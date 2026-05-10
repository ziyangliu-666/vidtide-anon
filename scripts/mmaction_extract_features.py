"""Extract pre-classifier features from a K400-pretrained mmaction2 backbone.

Runs in the `mmaction2` conda env (Python 3.10 + torch 2.1.0 + mmcv 2.1).
Reads ft_demamba_{train,test}.jsonl, writes data/ft_features/{backbone}_{X,y,ids}.

Supported backbones (--backbone):
  tsm      — Recognizer2D, 8 segments, 224x224
  i3d      — Recognizer3D, 32 frames, 224x224
  slowfast — Recognizer3D, 32 frames, 224x224
  swin     — Recognizer3D, 32 frames, 224x224 (VideoSwin-T)

Each video → 32 RGB frames (uint8 224x224 via existing ffmpeg helper) →
model.backbone(...) → adaptive_avg_pool to a fixed-dim vector → float16 row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.detection.dataset import extract_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mmext")

CKPT_DIR = REPO / "vendor" / "mmaction_ckpts"
SPLIT_DIR = REPO / "data" / "splits"
BLOB_ROOT = REPO / "data" / "blobs" / "videos"
OUT_DIR = REPO / "data" / "ft_features"

# ImageNet normalization (mmaction2 K400 configs use these by default for RGB models)
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)

BACKBONES = {
    "tsm": {
        "config": "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py",
        "ckpt_glob": "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_*.pth",
        "num_frames": 8,
        "is_3d": False,  # Recognizer2D path
    },
    "i3d": {
        "config": "i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py",
        "ckpt_glob": "i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_*.pth",
        "num_frames": 32,
        "is_3d": True,
    },
    "slowfast": {
        "config": "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py",
        "ckpt_glob": "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_*.pth",
        "num_frames": 32,
        "is_3d": True,
    },
    "swin": {
        "config": "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py",
        "ckpt_glob": "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_*.pth",
        "num_frames": 32,
        "is_3d": True,
    },
}


def load_split(name: str) -> list[dict]:
    rows = []
    with (SPLIT_DIR / f"ft_demamba_{name}.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def video_path(vid: str, label: int) -> Path:
    sub = "fake" if label == 1 else "real"
    return BLOB_ROOT / sub / f"{vid}.mp4"


def build_model(backbone: str, device: str):
    from mmaction.apis import init_recognizer
    spec = BACKBONES[backbone]
    cfg = CKPT_DIR / spec["config"]
    ckpt_candidates = list(CKPT_DIR.glob(spec["ckpt_glob"]))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No ckpt matching {spec['ckpt_glob']}")
    ckpt = ckpt_candidates[0]
    log.info("loading %s: %s", backbone, ckpt.name)
    model = init_recognizer(str(cfg), str(ckpt), device=device)
    model.eval()
    return model, spec


def extract_one(sample, num_frames):
    vid, label = sample
    p = video_path(vid, label)
    try:
        frames = extract_frames(p, num_frames=num_frames, resolution=224)
        return vid, label, frames, None
    except Exception as e:
        return vid, label, None, str(e)


def iter_samples(samples, num_frames, prefetch):
    if prefetch <= 0:
        for s in samples:
            yield extract_one(s, num_frames)
        return
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        yield from pool.map(lambda s: extract_one(s, num_frames), samples)


def featurize(model, frames: np.ndarray, spec: dict, mean: torch.Tensor, std: torch.Tensor, device: str) -> np.ndarray:
    """frames: (T, H, W, 3) uint8. Returns 1-D feature numpy array."""
    x = torch.from_numpy(frames).float().to(device)         # (T, H, W, 3)
    x = x.permute(0, 3, 1, 2).unsqueeze(0)                  # (1, T, 3, H, W)
    x = (x - mean) / std

    B, T = x.shape[0], x.shape[1]
    with torch.no_grad():
        if spec["is_3d"]:
            inp = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
            feat = model.backbone(inp)
        else:
            inp = x.view(B * T, 3, x.shape[-2], x.shape[-1])  # (B*T, C, H, W)
            feat = model.backbone(inp)

    if isinstance(feat, (tuple, list)) and len(feat) == 2 and all(t.dim() == 5 for t in feat):
        # SlowFast: (slow, fast) two streams, each (B, C, T', H', W')
        pooled = [F.adaptive_avg_pool3d(f, 1).flatten(1) for f in feat]
        v = torch.cat(pooled, dim=1)  # (B, C_slow + C_fast)
    else:
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.dim() == 5:
            v = F.adaptive_avg_pool3d(feat, 1).flatten(1)  # (B, C)
        elif feat.dim() == 4:
            if not spec["is_3d"]:
                # TSM: (B*T, C, H, W) → pool spatial → (B*T, C) → reshape (B, T, C) → mean over T
                p = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B*T, C)
                v = p.view(B, T, -1).mean(dim=1)  # (B, C)
            else:
                v = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        else:
            v = feat.view(B, -1)
    return v.squeeze(0).cpu().numpy().astype(np.float16)


def run_split(name: str, backbone: str, model, spec, device, mean, std, prefetch: int) -> None:
    rows = load_split(name)
    log.info("%s/%s: %d videos", backbone, name, len(rows))
    samples = [(r["video"], r["label"]) for r in rows]
    num_frames = spec["num_frames"]

    # Probe feature dim with first successful sample
    feat_dim = None
    feats_list = []
    labels = []
    ids = []

    t0 = time.time()
    errs = 0
    for i, (vid, label, frames, err) in enumerate(iter_samples(samples, num_frames, prefetch)):
        if err is not None:
            log.warning("frame err %s: %s", vid, err)
            errs += 1
            continue
        try:
            v = featurize(model, frames, spec, mean, std, device)
            if feat_dim is None:
                feat_dim = v.shape[0]
                log.info("%s feat_dim=%d", backbone, feat_dim)
            feats_list.append(v)
            labels.append(label)
            ids.append(vid)
        except Exception as e:
            log.warning("model err %s: %s", vid, e)
            errs += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(samples) - i - 1) / max(rate, 0.001)
            log.info("  [%d/%d] elapsed=%ds rate=%.2fv/s eta=%dm errs=%d",
                     i + 1, len(samples), int(elapsed), rate, int(eta / 60), errs)

    if not feats_list:
        log.error("%s/%s: no successful samples", backbone, name)
        return

    X = np.stack(feats_list, axis=0)
    y = np.asarray(labels, dtype=np.uint8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{backbone}_{name}_X.npy", X)
    np.save(OUT_DIR / f"{backbone}_{name}_y.npy", y)
    with (OUT_DIR / f"{backbone}_{name}_ids.txt").open("w") as f:
        for v in ids: f.write(v + "\n")
    log.info("%s/%s: wrote %d feats (dim=%d, %d errors), elapsed=%ds → %s",
             backbone, name, len(feats_list), X.shape[1], errs, int(time.time() - t0), OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES.keys()))
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("device=%s backbone=%s", device, args.backbone)
    model, spec = build_model(args.backbone, device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    for s in args.splits:
        run_split(s, args.backbone, model, spec, device, mean, std, args.prefetch)


if __name__ == "__main__":
    main()
