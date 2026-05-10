#!/usr/bin/env python3
"""Compute AUROC / bACC / F1 + per-generator + per-platform heatmaps
from cached detector scores.

Usage:
    python scripts/compute_metrics.py --detector clip_zero_shot --benchmark vidtide_m0
    python scripts/compute_metrics.py --all                              # table across all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.detection.metrics import all_metrics, per_group_heatmap


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load scores JSONL and return (scores, labels, generators, platforms)."""
    scores, labels, gens, plats = [], [], [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores.append(float(r["score"]))
            labels.append(1 if r.get("label") == "fake" else 0)
            gens.append(r.get("claimed_generator") or "unknown")
            plats.append(r.get("source_platform") or "unknown")
    return np.array(scores), np.array(labels), gens, plats


def print_row(detector: str, benchmark: str, metrics: dict) -> None:
    print(f"{detector:<20} {benchmark:<15} "
          f"AUROC={metrics['auroc']:.3f}  "
          f"bACC={metrics['bacc']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"(n_fake={metrics['n_fake']}, n_real={metrics['n_real']})")


def main():
    parser = argparse.ArgumentParser(description="Compute eval metrics")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--benchmark", default="vidtide_m0")
    parser.add_argument("--all", action="store_true", help="Across all cached detectors")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--heatmap", action="store_true", help="Print per-generator + per-platform breakdown")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"No results dir: {results_dir}")
        return

    if args.all:
        detectors = [d.name for d in results_dir.iterdir() if d.is_dir()]
    elif args.detector:
        detectors = [args.detector]
    else:
        print("Specify --detector NAME or --all")
        return

    print(f"{'Detector':<20} {'Benchmark':<15} Metrics")
    print("-" * 80)
    for det in detectors:
        path = results_dir / det / f"{args.benchmark}.jsonl"
        if not path.exists():
            print(f"{det:<20} {args.benchmark:<15} (no scores)")
            continue
        scores, labels, gens, plats = load_scores(path)
        if len(scores) == 0:
            print(f"{det:<20} {args.benchmark:<15} (empty)")
            continue
        m = all_metrics(scores, labels)
        print_row(det, args.benchmark, m)

        if args.heatmap:
            print(f"  Per-generator AUROC:")
            for g, a in sorted(per_group_heatmap(scores, labels, gens).items()):
                print(f"    {g:<30} {a:.3f}" if not np.isnan(a) else f"    {g:<30} n/a")
            print(f"  Per-platform AUROC:")
            for p, a in sorted(per_group_heatmap(scores, labels, plats).items()):
                print(f"    {p:<30} {a:.3f}" if not np.isnan(a) else f"    {p:<30} n/a")


if __name__ == "__main__":
    main()
