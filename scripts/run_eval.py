#!/usr/bin/env python3
"""Run a detector on the local VidTide dataset.

Usage:
    python scripts/run_eval.py --detector clip_zero_shot --benchmark vidtide_m0
    python scripts/run_eval.py --detector clip_zero_shot --limit 100
    python scripts/run_eval.py --detector clip_zero_shot --platform bilibili
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.detection.runner import EvalSpec, run_eval


def main():
    parser = argparse.ArgumentParser(description="Run a detector on VidTide")
    parser.add_argument("--detector", required=True, help="Detector name (e.g. clip_zero_shot)")
    parser.add_argument("--benchmark", default="vidtide_m0", help="vidtide_m0 | genvideo")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--platform", default=None, help="Filter: bilibili | reddit | ...")
    parser.add_argument("--db", default="data/vidtide.db")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    spec = EvalSpec(
        detector=args.detector,
        benchmark=args.benchmark,
        num_frames=args.num_frames,
        resolution=args.resolution,
        limit=args.limit,
        platform_filter=args.platform,
    )
    out = run_eval(spec, Path(args.db), Path(args.results_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
