"""Evaluate the saved mmaction LP head on a fresh OOD set (data/ood_fresh/).

Runs in the `mmaction2` conda env (same as scripts/mmaction_extract_features.py).
Reads data/ood_fresh/ood_fresh.jsonl, extracts features for each downloaded
video, applies the saved {backbone}_lp_fc1.pt head, and reports AUROC
overall + per-platform + per-generator.

Usage:
  python scripts/ood_eval_lp.py --backbone slowfast
  python scripts/ood_eval_lp.py --backbone slowfast --backbone i3d ...

Outputs:
  results/ood_fresh_{backbone}.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.mmaction_extract_features import (  # noqa: E402
    BACKBONES, IMAGENET_MEAN, IMAGENET_STD, build_model, featurize, iter_samples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ood")

OOD_ROOT = REPO / "data" / "ood_fresh"
MANIFEST = OOD_ROOT / "ood_fresh.jsonl"
FEAT_DIR = REPO / "data" / "ft_features"
RESULTS = REPO / "results"


def auroc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return float("nan")
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    tpr = np.concatenate(([0.0], tp / P))
    fpr = np.concatenate(([0.0], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def load_manifest():
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("status") != "done": continue
        rows.append(r)
    return rows


def video_path(row):
    sub = "fake" if row["label"] == 1 else "real"
    return OOD_ROOT / sub / f"{row['video']}.mp4"


def extract_features(backbone, rows, device, prefetch=4):
    model, spec = build_model(backbone, device)
    mean = IMAGENET_MEAN.to(device); std = IMAGENET_STD.to(device)
    samples = [(r["video"], r["label"], video_path(r)) for r in rows]
    num_frames = spec["num_frames"]

    # Reuse iter_samples which expects (vid, label) and uses its own path
    # helper. Inline the extraction instead to pass explicit paths.
    from concurrent.futures import ThreadPoolExecutor
    from server.detection.dataset import extract_frames

    def _one(sample):
        vid, label, p = sample
        try:
            fr = extract_frames(p, num_frames=num_frames, resolution=224)
            return vid, label, fr, None
        except Exception as e:
            return vid, label, None, str(e)

    feats = []; out_rows = []
    t0 = time.time()
    errs = 0
    with ThreadPoolExecutor(max_workers=max(1, prefetch)) as pool:
        for i, (vid, label, fr, err) in enumerate(pool.map(_one, samples), 1):
            if err is not None:
                log.warning("frame err %s: %s", vid, err); errs += 1; continue
            try:
                v = featurize(model, fr, spec, mean, std, device)
                feats.append(v)
                out_rows.append(next(r for r in rows if r["video"] == vid))
            except Exception as e:
                log.warning("model err %s: %s", vid, e); errs += 1
            if i % 10 == 0:
                log.info("  [%d/%d] elapsed=%ds errs=%d", i, len(samples), int(time.time()-t0), errs)
    X = np.stack(feats, axis=0)
    y = np.array([r["label"] for r in out_rows], dtype=np.int64)
    log.info("%s OOD: %d feats dim=%d errs=%d", backbone, len(out_rows), X.shape[1], errs)
    return X, y, out_rows


def report(backbone, X, y, rows):
    sd = torch.load(FEAT_DIR / f"{backbone}_lp_fc1.pt", map_location="cpu")
    W = sd["weight"].numpy().squeeze().astype(np.float32)
    b = float(sd["bias"].numpy().squeeze())
    scores = X.astype(np.float32) @ W + b

    overall = auroc(scores, y)

    by_platform = defaultdict(lambda: ([], []))
    by_gen = defaultdict(lambda: ([], []))
    for s, yi, r in zip(scores, y, rows):
        by_platform[r["platform"]][0].append(s); by_platform[r["platform"]][1].append(yi)
        if yi == 1:
            by_gen[r.get("generator") or "unk"][0].append(s)

    # For per-generator AUROC use each fake-gen-subset + full real pool as neg
    real_scores = [s for s, yi in zip(scores, y) if yi == 0]

    plat_rows = []
    for p, (ss, ys) in sorted(by_platform.items()):
        n = len(ys); npos = int(np.sum(ys)); nneg = n - npos
        a = auroc(ss, ys) if npos and nneg else float("nan")
        plat_rows.append({"platform": p, "n": n, "pos": npos, "neg": nneg,
                          "AUROC": round(a, 4) if a == a else None,
                          "mean_score": round(float(np.mean(ss)), 4)})
    gen_rows = []
    for g, (ss, _) in sorted(by_gen.items(), key=lambda x: -len(x[1][0])):
        gs = np.concatenate([ss, real_scores])
        gy = np.concatenate([np.ones(len(ss)), np.zeros(len(real_scores))])
        gen_rows.append({"generator": g, "n": len(ss),
                         "AUROC_vs_real_pool": round(auroc(gs, gy), 4),
                         "mean_score": round(float(np.mean(ss)), 4)})

    out = {
        "backbone": backbone,
        "n": int(len(y)),
        "n_fake": int(np.sum(y == 1)),
        "n_real": int(np.sum(y == 0)),
        "overall_AUROC": round(overall, 4),
        "score_stats": {
            "real_mean": round(float(np.mean([s for s, yi in zip(scores, y) if yi == 0])), 4),
            "real_std": round(float(np.std([s for s, yi in zip(scores, y) if yi == 0])), 4),
            "fake_mean": round(float(np.mean([s for s, yi in zip(scores, y) if yi == 1])), 4),
            "fake_std": round(float(np.std([s for s, yi in zip(scores, y) if yi == 1])), 4),
        },
        "by_platform": plat_rows,
        "by_generator": gen_rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", action="append", default=None,
                    help="Backbone name (can pass multiple). Default: slowfast only.")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    backbones = args.backbone or ["slowfast"]
    device = args.device if torch.cuda.is_available() else "cpu"

    rows = load_manifest()
    log.info("OOD manifest: %d successful rows", len(rows))
    if not rows:
        log.error("no rows found; run scripts/download_ood_fresh.py first"); return 1

    RESULTS.mkdir(exist_ok=True)
    for bb in backbones:
        if bb not in BACKBONES:
            log.error("unknown backbone %s (choose from %s)", bb, list(BACKBONES))
            continue
        head_path = FEAT_DIR / f"{bb}_lp_fc1.pt"
        if not head_path.exists():
            log.error("no saved head: %s", head_path); continue
        X, y, kept = extract_features(bb, rows, device, prefetch=args.prefetch)
        # cache OOD features
        np.save(FEAT_DIR / f"{bb}_ood_X.npy", X)
        np.save(FEAT_DIR / f"{bb}_ood_y.npy", y)
        with (FEAT_DIR / f"{bb}_ood_ids.txt").open("w") as f:
            for r in kept: f.write(r["video"] + "\n")
        rep = report(bb, X, y, kept)
        out_path = RESULTS / f"ood_fresh_{bb}.json"
        out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        log.info("%s: overall AUROC = %.4f (n=%d)  → %s",
                 bb, rep["overall_AUROC"], rep["n"], out_path)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
