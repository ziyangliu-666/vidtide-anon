"""Cleanlab label-error audit on the 5K bench scores.

Pools per-video scores from all 9 detector runs, ensembles them via mean rank,
and runs cleanlab.classification.find_label_issues on (label, ensemble_proba)
to surface platform-tag errors.

Output:
  - results/cleanlab_label_issues.json  (top-K most suspicious videos)
  - results/cleanlab_label_issues.md     (human-reviewable list with platform/generator)
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "vidtide.db"
RESULTS_DIR = REPO / "results"
TOP_K = 100


def load_db_meta() -> dict[str, dict]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator, ''), '') AS gen, "
        "label, label_source, COALESCE(label_confidence, 0) FROM videos"
    ).fetchall()
    con.close()
    out = {}
    for vid_id, plat, gen, lab, lsrc, lconf in rows:
        out[f"{vid_id}.mp4"] = {
            "platform": plat, "generator": gen or None,
            "db_label": lab, "label_source": lsrc, "label_conf": lconf,
        }
    return out


def load_all_scores() -> dict[str, dict]:
    """Returns {video: {detector: score, label: int}}."""
    by_video: dict[str, dict] = defaultdict(dict)
    for p in sorted(RESULTS_DIR.glob("bench_5k_*_scores.jsonl")):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                v = r["video"]
                by_video[v][r["detector"]] = float(r["score"])
                by_video[v]["label"] = int(r["label"])
    return dict(by_video)


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to [0,1] via rank — robust to per-detector scale differences."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    return ranks / max(len(scores) - 1, 1)


def main() -> None:
    from cleanlab.filter import find_label_issues

    meta = load_db_meta()
    vids = load_all_scores()
    print(f"Loaded {len(vids)} videos with at least one detector score")

    detector_names = sorted({d for v in vids.values() for d in v if d != "label"})
    print(f"Detectors found: {detector_names}")

    # Build (n_videos, n_detectors) score matrix; videos with missing detectors filled with NaN.
    video_list = sorted(vids.keys())
    raw_mat = np.full((len(video_list), len(detector_names)), np.nan)
    labels = np.zeros(len(video_list), dtype=int)
    for i, v in enumerate(video_list):
        labels[i] = vids[v]["label"]
        for j, d in enumerate(detector_names):
            if d in vids[v]:
                raw_mat[i, j] = vids[v][d]

    # Per-detector rank-normalize over the videos that have a score
    norm_mat = np.full_like(raw_mat, np.nan)
    for j in range(raw_mat.shape[1]):
        col = raw_mat[:, j]
        m = ~np.isnan(col)
        if m.sum() > 0:
            norm_mat[m, j] = rank_normalize(col[m])

    # Ensemble: mean of valid normalized ranks → AI-prob in [0,1]
    ensemble = np.nanmean(norm_mat, axis=1)
    keep = ~np.isnan(ensemble)
    ensemble = ensemble[keep]
    labels = labels[keep]
    video_list = [v for v, k in zip(video_list, keep) if k]
    print(f"After filtering: {len(video_list)} videos with ensemble score")

    # cleanlab needs (n, n_classes) class probabilities
    pred_proba = np.column_stack([1 - ensemble, ensemble])  # P(real), P(fake)

    # Find label issues (default: prune_by_noise_rate strategy, ranked by self-confidence)
    issues_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_proba,
        return_indices_ranked_by="self_confidence",
    )
    print(f"cleanlab flagged {len(issues_idx)} potential label errors "
          f"({100 * len(issues_idx) / len(labels):.1f}% of pool)")

    # Top-K most suspicious
    top = issues_idx[:TOP_K]
    suspects = []
    for rank, idx in enumerate(top, 1):
        v = video_list[idx]
        m = meta.get(v, {})
        suspects.append({
            "rank": rank,
            "video": v,
            "platform_label": "fake" if labels[idx] == 1 else "real",
            "ensemble_ai_prob": round(float(ensemble[idx]), 4),
            "platform": m.get("platform"),
            "generator": m.get("generator"),
            "label_source": m.get("label_source"),
            "label_conf": m.get("label_conf"),
        })

    out_json = RESULTS_DIR / "cleanlab_label_issues.json"
    with out_json.open("w") as f:
        json.dump({
            "n_videos_audited": len(labels),
            "n_flagged": int(len(issues_idx)),
            "flag_rate": round(len(issues_idx) / len(labels), 4),
            "n_detectors_in_ensemble": int(raw_mat.shape[1]),
            "top_k": TOP_K,
            "suspects": suspects,
        }, f, indent=2)
    print(f"→ {out_json}")

    # Markdown summary
    out_md = RESULTS_DIR / "cleanlab_label_issues.md"
    with out_md.open("w") as f:
        f.write(f"# Cleanlab label-error audit\n\n")
        f.write(f"- Pool: {len(labels)} videos with ≥1 detector score (5K bench output)\n")
        f.write(f"- Detectors in ensemble: {raw_mat.shape[1]}\n")
        f.write(f"- Flagged as potential label errors: {len(issues_idx)} ({100 * len(issues_idx) / len(labels):.1f}%)\n")
        f.write(f"- Method: rank-normalized ensemble of detector scores → cleanlab.find_label_issues "
                f"(self_confidence ranking)\n\n")
        f.write(f"## By platform label\n\n")
        n_fake_flag = sum(1 for s in suspects if s["platform_label"] == "fake")
        n_real_flag = TOP_K - n_fake_flag
        f.write(f"In the top-{TOP_K} most suspicious:\n")
        f.write(f"- {n_fake_flag} are platform-labeled **fake** but ensemble thinks **real** (low ai_prob)\n")
        f.write(f"- {n_real_flag} are platform-labeled **real** but ensemble thinks **fake** (high ai_prob)\n\n")
        f.write(f"## Top-{TOP_K} suspects (manual review needed)\n\n")
        f.write("| rank | video | label | ai_prob | platform | generator | label_source |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in suspects:
            gen = s["generator"] or "-"
            lsrc = s["label_source"] or "-"
            f.write(f"| {s['rank']} | `{s['video']}` | {s['platform_label']} | "
                    f"{s['ensemble_ai_prob']} | {s['platform']} | {gen} | {lsrc} |\n")
    print(f"→ {out_md}")


if __name__ == "__main__":
    main()
