"""Hard-case analysis on the 5K bench scores.

For each video, compute the ensemble (rank-normalized) AI-probability across
all detectors. A "hard case" is a video where mean confidence in the correct
class is < 0.6 (i.e. detectors are confused).

Outputs:
  - results/hard_cases.json (top-200 hardest fakes + top-200 hardest reals)
  - results/hard_cases.md (counts by platform/generator + sample list)
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "vidtide.db"
RESULTS_DIR = REPO / "results"


def load_db_meta() -> dict[str, dict]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator, ''), '') AS gen, "
        "label, label_source FROM videos"
    ).fetchall()
    con.close()
    return {f"{vid}.mp4": {"plat": p, "gen": g or None, "label": lab, "lsrc": ls}
            for vid, p, g, lab, ls in rows}


def rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def main() -> None:
    meta = load_db_meta()

    # video → {detector: score, label: int}
    by_video: dict[str, dict] = defaultdict(dict)
    for p in sorted(RESULTS_DIR.glob("bench_5k_*_scores.jsonl")):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                by_video[r["video"]][r["detector"]] = float(r["score"])
                by_video[r["video"]]["label"] = int(r["label"])

    detectors = sorted({d for v in by_video.values() for d in v if d != "label"})
    vids = sorted(by_video.keys())
    raw = np.full((len(vids), len(detectors)), np.nan)
    labels = np.zeros(len(vids), dtype=int)
    for i, v in enumerate(vids):
        labels[i] = by_video[v]["label"]
        for j, d in enumerate(detectors):
            if d in by_video[v]:
                raw[i, j] = by_video[v][d]

    # Per-detector rank-normalize
    norm = np.full_like(raw, np.nan)
    for j in range(raw.shape[1]):
        m = ~np.isnan(raw[:, j])
        if m.sum() > 0:
            norm[m, j] = rank_norm(raw[m, j])

    # Ensemble AI-prob
    ens_ai = np.nanmean(norm, axis=1)
    keep = ~np.isnan(ens_ai)
    ens_ai = ens_ai[keep]
    labels = labels[keep]
    vids = [v for v, k in zip(vids, keep) if k]
    n_det = (~np.isnan(norm)).sum(axis=1)[keep]

    # Confidence-in-correct-class: for fakes, ai_prob is correct; for reals, 1 - ai_prob
    conf = np.where(labels == 1, ens_ai, 1 - ens_ai)
    is_hard = conf < 0.6

    # Stratify hards
    fake_hard_idx = np.where((labels == 1) & is_hard)[0]
    real_hard_idx = np.where((labels == 0) & is_hard)[0]

    # Sort by confidence ascending (hardest first)
    fake_hard_sorted = fake_hard_idx[np.argsort(conf[fake_hard_idx])]
    real_hard_sorted = real_hard_idx[np.argsort(conf[real_hard_idx])]

    # Count by platform / generator for fakes
    fake_plat = Counter()
    fake_gen = Counter()
    fake_total_plat = Counter()
    fake_total_gen = Counter()
    for i, v in enumerate(vids):
        m = meta.get(v)
        if not m or labels[i] != 1:
            continue
        fake_total_plat[m["plat"]] += 1
        fake_total_gen[m["gen"] or "_unlabeled"] += 1
        if is_hard[i]:
            fake_plat[m["plat"]] += 1
            fake_gen[m["gen"] or "_unlabeled"] += 1

    real_plat = Counter()
    real_total_plat = Counter()
    for i, v in enumerate(vids):
        m = meta.get(v)
        if not m or labels[i] != 0:
            continue
        real_total_plat[m["plat"]] += 1
        if is_hard[i]:
            real_plat[m["plat"]] += 1

    out = {
        "n_videos": int(keep.sum()),
        "n_fake": int((labels == 1).sum()),
        "n_real": int((labels == 0).sum()),
        "n_hard_fake": int(len(fake_hard_idx)),
        "n_hard_real": int(len(real_hard_idx)),
        "hard_threshold_conf": 0.6,
        "fake_hard_by_platform": {
            k: {"hard": fake_plat[k], "total": fake_total_plat[k],
                "rate": round(fake_plat[k] / max(fake_total_plat[k], 1), 4)}
            for k in fake_total_plat
        },
        "fake_hard_by_generator": {
            k: {"hard": fake_gen[k], "total": fake_total_gen[k],
                "rate": round(fake_gen[k] / max(fake_total_gen[k], 1), 4)}
            for k in fake_total_gen if fake_total_gen[k] >= 50
        },
        "real_hard_by_platform": {
            k: {"hard": real_plat[k], "total": real_total_plat[k],
                "rate": round(real_plat[k] / max(real_total_plat[k], 1), 4)}
            for k in real_total_plat
        },
        "top_hard_fakes": [
            {"video": vids[i], "ensemble_ai_prob": round(float(ens_ai[i]), 4),
             "n_detectors": int(n_det[i]),
             "platform": meta.get(vids[i], {}).get("plat"),
             "generator": meta.get(vids[i], {}).get("gen"),
             "label_source": meta.get(vids[i], {}).get("lsrc")}
            for i in fake_hard_sorted[:200]
        ],
        "top_hard_reals": [
            {"video": vids[i], "ensemble_ai_prob": round(float(ens_ai[i]), 4),
             "n_detectors": int(n_det[i]),
             "platform": meta.get(vids[i], {}).get("plat")}
            for i in real_hard_sorted[:200]
        ],
    }
    out_path = RESULTS_DIR / "hard_cases.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"→ {out_path}")

    # Markdown
    md = RESULTS_DIR / "hard_cases.md"
    with md.open("w") as f:
        f.write("# Hard-case analysis (ensemble confidence < 0.6)\n\n")
        f.write(f"- Audited: {out['n_videos']} videos ({out['n_fake']} fake / {out['n_real']} real)\n")
        f.write(f"- Hard fakes: {out['n_hard_fake']} ({100*out['n_hard_fake']/max(out['n_fake'],1):.1f}%)\n")
        f.write(f"- Hard reals: {out['n_hard_real']} ({100*out['n_hard_real']/max(out['n_real'],1):.1f}%)\n\n")
        f.write("## Fake hard-rate by platform\n\n| platform | hard | total | rate |\n|---|---|---|---|\n")
        for k, v in sorted(out["fake_hard_by_platform"].items(), key=lambda kv: -kv[1]["rate"]):
            f.write(f"| {k} | {v['hard']} | {v['total']} | {v['rate']:.2%} |\n")
        f.write("\n## Fake hard-rate by generator (≥50 fakes)\n\n| generator | hard | total | rate |\n|---|---|---|---|\n")
        for k, v in sorted(out["fake_hard_by_generator"].items(), key=lambda kv: -kv[1]["rate"]):
            f.write(f"| {k} | {v['hard']} | {v['total']} | {v['rate']:.2%} |\n")
        f.write("\n## Real hard-rate by platform\n\n| platform | hard | total | rate |\n|---|---|---|---|\n")
        for k, v in sorted(out["real_hard_by_platform"].items(), key=lambda kv: -kv[1]["rate"]):
            f.write(f"| {k} | {v['hard']} | {v['total']} | {v['rate']:.2%} |\n")
        f.write("\n## Top-30 hardest fakes (most-confidently-misclassified)\n\n| video | ai_prob | n_det | platform | generator | label_src |\n|---|---|---|---|---|---|\n")
        for s in out["top_hard_fakes"][:30]:
            f.write(f"| `{s['video']}` | {s['ensemble_ai_prob']} | {s['n_detectors']} | "
                    f"{s['platform']} | {s['generator'] or '-'} | {s['label_source'] or '-'} |\n")
    print(f"→ {md}")


if __name__ == "__main__":
    main()
