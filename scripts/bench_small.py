"""scripts/bench_small.py — quick AUROC bench of registered detectors on local fake/real videos.

Usage:
    python scripts/bench_small.py --detectors npr_pika tall_pika --n-per-class 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.detection.dataset import extract_frames  # noqa: E402
from server.detection.metrics import all_metrics  # noqa: E402
from server.detection.registry import load_detector  # noqa: E402


def _extract_one(sample: tuple[Path, int], num_frames: int, resolution: int):
    path, label = sample
    try:
        frames = extract_frames(path, num_frames=num_frames, resolution=resolution)
        return path, label, frames, None
    except Exception as e:
        return path, label, None, str(e)


def _iter_samples(samples, num_frames, resolution, prefetch):
    if prefetch <= 0:
        for s in samples:
            yield _extract_one(s, num_frames, resolution)
        return
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        yield from pool.map(
            lambda s: _extract_one(s, num_frames, resolution), samples
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bench_small")


def collect(label_dir: Path, n: int, seed: int, exclude: set[str] | None = None) -> list[Path]:
    files = sorted(label_dir.glob("*.mp4"))
    if exclude:
        files = [f for f in files if str(f) not in exclude]
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True,
                    help="Names registered in server.detection.detectors.* (e.g. npr_pika tall_pika)")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--video-root", default="data/blobs/videos")
    ap.add_argument("--out", default="results/bench_small.json")
    ap.add_argument("--scores-out", default="results/bench_small_scores.jsonl",
                    help="Per-video scores written for diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--prefetch", type=int, default=0,
                    help="N>0 enables N-thread parallel frame extraction (predict stays on main thread).")
    args = ap.parse_args()

    fake_root = Path(args.video_root) / "fake"
    real_root = Path(args.video_root) / "real"

    # Reserve last 100 sorted reals as NSG-VD reference set (excluded from test pool).
    nsgvd_in_use = any(d.startswith("nsgvd_") for d in args.detectors)
    nsgvd_ref_reals: list[str] = []
    excluded_reals: set[str] = set()
    if nsgvd_in_use:
        all_reals_sorted = sorted(real_root.glob("*.mp4"))
        nsgvd_ref_reals = [str(p) for p in all_reals_sorted[-100:]]
        excluded_reals = set(nsgvd_ref_reals)
        logger.info("NSG-VD detected: reserved last %d reals as MMD ref set", len(nsgvd_ref_reals))

    fakes = collect(fake_root, args.n_per_class, args.seed)
    reals = collect(real_root, args.n_per_class, args.seed, exclude=excluded_reals)
    logger.info("Sampled %d fake + %d real from %s", len(fakes), len(reals), args.video_root)

    samples: list[tuple[Path, int]] = [(p, 1) for p in fakes] + [(p, 0) for p in reals]

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_f = open(scores_path, "w")

    overall: dict[str, dict] = {}
    for det_name in args.detectors:
        logger.info("=== %s ===", det_name)
        det_kwargs: dict = {}
        if det_name.startswith("nsgvd_"):
            det_kwargs["ref_videos"] = nsgvd_ref_reals
        det = load_detector(det_name, **det_kwargs)
        scores: list[float] = []
        labels: list[int] = []
        errs = 0
        t0 = time.time()
        iterator = _iter_samples(samples, args.num_frames, args.resolution, args.prefetch)
        for i, (path, label, frames, extract_err) in enumerate(iterator, 1):
            if extract_err is not None:
                logger.warning("err on %s: %s", path.name, extract_err)
                errs += 1
                continue
            try:
                score = det.predict(frames)
            except Exception as e:
                logger.warning("err on %s: %s", path.name, e)
                errs += 1
                continue
            scores.append(score)
            labels.append(label)
            scores_f.write(json.dumps({
                "detector": det_name,
                "video": path.name,
                "label": label,
                "score": float(score),
            }) + "\n")
            scores_f.flush()
            if i % 25 == 0:
                logger.info("  [%d/%d] elapsed=%ds errs=%d", i, len(samples), int(time.time() - t0), errs)
        det.close()

        m = all_metrics(np.array(scores), np.array(labels))
        m["errors"] = errs
        m["wall_seconds"] = round(time.time() - t0, 1)
        overall[det_name] = m
        logger.info(
            "%s: AUROC=%.4f bACC=%.4f F1=%.4f errs=%d %.1fs",
            det_name, m["auroc"], m["bacc"], m["f1"], errs, m["wall_seconds"],
        )

    scores_f.close()
    with open(args.out, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("=" * 80)
    print(f"{'detector':<20} {'AUROC':<8} {'bACC':<8} {'F1':<8} {'n_fake':<7} {'n_real':<7} {'errs':<6} time")
    print("-" * 80)
    for name, m in overall.items():
        print(
            f"{name:<20} {m['auroc']:<8.4f} {m['bacc']:<8.4f} {m['f1']:<8.4f} "
            f"{m['n_fake']:<7} {m['n_real']:<7} {m['errors']:<6} {m['wall_seconds']}s"
        )
    print()
    print(f"results → {args.out}")
    print(f"scores  → {scores_path}")


if __name__ == "__main__":
    main()
