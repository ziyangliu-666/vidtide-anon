"""Slice 5K bench scores by generator and platform.

Joins per-video scores from results/bench_5k_*_scores.jsonl with vidtide.db
on filename → (source_platform, claimed_generator). Computes per-slice AUROC
for each (detector, group) pair where the slice has ≥ MIN_FAKE labeled fakes.

Real samples are shared across all generator slices (same neg pool); per-platform
slices use that platform's reals when available, else fall back to all reals.

Outputs:
  - results/per_generator_auroc.json
  - results/per_platform_auroc.json
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "vidtide.db"
SCORES_GLOB = "bench_5k_*_scores.jsonl"
RESULTS_DIR = REPO / "results"

MIN_FAKE = 50  # min fake samples in a slice to report AUROC


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal-area AUROC (handles ties via descending sort)."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate(([0.0], tp / P))
    fpr = np.concatenate(([0.0], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def load_db_meta() -> dict[str, tuple[str, str | None]]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator, ''), '') "
        "FROM videos"
    ).fetchall()
    con.close()
    out: dict[str, tuple[str, str | None]] = {}
    for vid_id, plat, gen in rows:
        out[f"{vid_id}.mp4"] = (plat, gen or None)
    return out


def load_scores() -> dict[str, list[dict]]:
    """Returns {detector: [{video, label, score}, ...]}."""
    rows: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(RESULTS_DIR.glob(SCORES_GLOB)):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                rows[r["detector"]].append(
                    {"video": r["video"], "label": int(r["label"]), "score": float(r["score"])}
                )
    return dict(rows)


def main() -> None:
    meta = load_db_meta()
    all_scores = load_scores()

    detectors = sorted(all_scores.keys())
    print(f"Loaded {len(detectors)} detectors, DB has {len(meta)} videos")

    # ------------ Per-generator slicing ------------
    # For each detector: { generator: AUROC, n_fake: int }
    # Reals are shared (full real pool used as negatives for every gen slice).
    per_gen: dict[str, dict[str, dict]] = {}
    gen_counts: dict[str, int] = defaultdict(int)

    for det in detectors:
        rows = all_scores[det]
        # reals: all label=0 in this detector's run
        real_scores = np.array([r["score"] for r in rows if r["label"] == 0])
        # fakes grouped by generator
        gen_to_fakes: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["label"] != 1:
                continue
            m = meta.get(r["video"])
            if m is None:
                continue
            gen = m[1]
            if gen is None:
                gen = "_unlabeled"
            gen_to_fakes[gen].append(r["score"])

        det_out: dict[str, dict] = {}
        for gen, fakes in sorted(gen_to_fakes.items(), key=lambda kv: -len(kv[1])):
            if len(fakes) < MIN_FAKE:
                continue
            scores = np.concatenate([np.asarray(fakes), real_scores])
            labels = np.concatenate([np.ones(len(fakes)), np.zeros(len(real_scores))])
            det_out[gen] = {
                "auroc": round(auroc(scores, labels), 4),
                "n_fake": len(fakes),
                "n_real": len(real_scores),
            }
            gen_counts[gen] = max(gen_counts[gen], len(fakes))
        per_gen[det] = det_out
        print(f"  {det}: {len(det_out)} generator slices")

    # ------------ Per-platform slicing ------------
    # For each detector & platform: use that platform's reals if present, else
    # fall back to global real pool.
    per_plat: dict[str, dict[str, dict]] = {}
    plat_counts_fake: dict[str, int] = defaultdict(int)
    plat_counts_real: dict[str, int] = defaultdict(int)

    for det in detectors:
        rows = all_scores[det]
        plat_to_fakes: dict[str, list[float]] = defaultdict(list)
        plat_to_reals: dict[str, list[float]] = defaultdict(list)
        all_reals: list[float] = []
        for r in rows:
            m = meta.get(r["video"])
            if m is None:
                continue
            plat = m[0]
            if r["label"] == 1:
                plat_to_fakes[plat].append(r["score"])
            else:
                plat_to_reals[plat].append(r["score"])
                all_reals.append(r["score"])

        det_out: dict[str, dict] = {}
        for plat, fakes in sorted(plat_to_fakes.items(), key=lambda kv: -len(kv[1])):
            if len(fakes) < MIN_FAKE:
                continue
            reals = plat_to_reals.get(plat) or all_reals
            real_src = "same-plat" if plat_to_reals.get(plat) else "global"
            scores = np.concatenate([np.asarray(fakes), np.asarray(reals)])
            labels = np.concatenate([np.ones(len(fakes)), np.zeros(len(reals))])
            det_out[plat] = {
                "auroc": round(auroc(scores, labels), 4),
                "n_fake": len(fakes),
                "n_real": len(reals),
                "real_src": real_src,
            }
            plat_counts_fake[plat] = max(plat_counts_fake[plat], len(fakes))
            plat_counts_real[plat] = max(plat_counts_real[plat], len(reals))
        per_plat[det] = det_out
        print(f"  {det}: {len(det_out)} platform slices")

    # ------------ Write outputs ------------
    gen_path = RESULTS_DIR / "per_generator_auroc.json"
    plat_path = RESULTS_DIR / "per_platform_auroc.json"
    with gen_path.open("w") as f:
        json.dump(
            {
                "min_fake": MIN_FAKE,
                "generator_max_n_fake": dict(gen_counts),
                "results": per_gen,
            },
            f, indent=2,
        )
    with plat_path.open("w") as f:
        json.dump(
            {
                "min_fake": MIN_FAKE,
                "platform_max_n_fake": dict(plat_counts_fake),
                "platform_max_n_real": dict(plat_counts_real),
                "results": per_plat,
            },
            f, indent=2,
        )
    print(f"\n→ {gen_path}")
    print(f"→ {plat_path}")

    # ------------ Print summary tables ------------
    # Generator table: detectors as rows, top-N generators as cols
    top_gens = sorted(gen_counts.items(), key=lambda kv: -kv[1])[:8]
    top_gen_names = [g for g, _ in top_gens if g != "_unlabeled"][:7]
    print(f"\n=== Per-generator AUROC ({len(top_gen_names)} top generators with ≥{MIN_FAKE} fakes per detector) ===")
    header = f"{'detector':<20} | " + " | ".join(f"{g[:9]:<9}" for g in top_gen_names) + f" | {'unlbl':<9}"
    print(header)
    print("-" * len(header))
    for det in detectors:
        cells = []
        for g in top_gen_names:
            v = per_gen[det].get(g, {}).get("auroc")
            cells.append(f"{v:<9.4f}" if v is not None else f"{'-':<9}")
        unl = per_gen[det].get("_unlabeled", {}).get("auroc")
        cells.append(f"{unl:<9.4f}" if unl is not None else f"{'-':<9}")
        print(f"{det:<20} | " + " | ".join(cells))

    # Platform table
    plat_names = ["bilibili", "reddit", "youtube", "showcase"]
    print(f"\n=== Per-platform AUROC (≥{MIN_FAKE} fakes per detector × platform) ===")
    header = f"{'detector':<20} | " + " | ".join(f"{p:<9}" for p in plat_names)
    print(header)
    print("-" * len(header))
    for det in detectors:
        cells = []
        for p in plat_names:
            v = per_plat[det].get(p, {}).get("auroc")
            cells.append(f"{v:<9.4f}" if v is not None else f"{'-':<9}")
        print(f"{det:<20} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
