"""Extract NSG-VD velocity features for VidTide FT split.

Per video: load 8 frames @ 224, run frozen diffusion model to get score,
compute NSG velocity tensor (T, 3, 224, 224), save to disk as fp32 .npy.

Run separately for --split train (8K) and --split test (2K). Resumable:
skips videos whose .npy already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nsgvd_feat")

REPO = Path(__file__).resolve().parent.parent
NSGVD_DIR = REPO / "vendor" / "NSG-VD"
DIFFUSION_CKPT = REPO / "Checkpoints" / "256x256_diffusion_uncond.pt"
SPLITS_DIR = REPO / "data" / "splits"
CACHE_ROOT = REPO / "data" / "ft_features" / "nsgvd"
BLOB_ROOT = REPO / "data" / "blobs" / "videos"

DIFFUSE_STEPS = 5
NUM_FRAMES = 8
RES = 224


def _setup_score_fn(device: str):
    """Load NSG-VD's frozen diffusion score function (uses chdir trick)."""
    if str(NSGVD_DIR) not in sys.path:
        sys.path.insert(0, str(NSGVD_DIR))
    if not DIFFUSION_CKPT.exists():
        raise FileNotFoundError(f"Diffusion ckpt missing: {DIFFUSION_CKPT}")
    cwd = os.getcwd()
    os.chdir(str(NSGVD_DIR))
    try:
        from data.utils import get_score_fn
        return get_score_fn(device=device, process_shape=(3, RES, RES))
    finally:
        os.chdir(cwd)


@torch.no_grad()
def extract_velocity(score_fn, frames: np.ndarray, device: str) -> np.ndarray:
    """frames (T,H,W,3) uint8 -> velocity (T, 3, RES, RES) fp32."""
    x = torch.from_numpy(frames).float().to(device) / 255.0
    x = x.permute(0, 3, 1, 2)  # (T, 3, H, W)
    if x.shape[-1] != RES or x.shape[-2] != RES:
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    T = x.shape[0]

    t_value = (DIFFUSE_STEPS + 1) / 1000
    curr_t = torch.tensor(t_value, device=device).expand(T)
    score_full = score_fn(2 * x - 1, curr_t)
    score, _ = torch.split(score_full, score_full.shape[1] // 2, dim=1)
    if score.shape[-1] != RES or score.shape[-2] != RES:
        score = F.interpolate(score, size=(RES, RES), mode="bilinear", align_corners=False)

    image = x.unsqueeze(0)  # (1, T, 3, RES, RES)
    score = score.view(1, T, 3, RES, RES)
    pixel_diff = torch.zeros_like(image)
    for t in range(1, T - 1):
        pixel_diff[:, t] = (image[:, t + 1] - image[:, t - 1]) / 2
    pixel_diff[:, 0] = image[:, 1] - image[:, 0]
    pixel_diff[:, -1] = image[:, -1] - image[:, -2]

    eps = 1e-10
    denom = (score * pixel_diff).sum(dim=(2, 3, 4)).view(1, T, 1, 1, 1) + eps
    velocity = (score / denom).squeeze(0).cpu().numpy().astype(np.float32)
    return velocity  # (T, 3, RES, RES)


def iter_split(split: str) -> Iterable[dict]:
    fpath = SPLITS_DIR / f"ft_demamba_{split}.jsonl"
    with fpath.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def video_path(rec: dict) -> Path:
    """Map jsonl record → on-disk video path."""
    sub = "fake" if rec["label"] == 1 else "real"
    return BLOB_ROOT / sub / f"{rec['video']}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="cap N videos (debug)")
    args = ap.parse_args()

    out_dir = CACHE_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_jsonl = CACHE_ROOT / f"{args.split}_index.jsonl"

    from server.detection.dataset import extract_frames

    score_fn = _setup_score_fn(args.device)

    records = list(iter_split(args.split))
    if args.limit:
        records = records[: args.limit]
    log.info("split=%s total=%d cache=%s", args.split, len(records), out_dir)

    n_done = n_skip = n_err = 0
    written_index: list[dict] = []
    for i, rec in enumerate(records):
        vid = rec["video"]
        out_path = out_dir / f"{vid}.npy"
        if out_path.exists():
            written_index.append(rec)
            n_skip += 1
            continue
        vp = video_path(rec)
        if not vp.exists():
            n_err += 1
            continue
        try:
            frames = extract_frames(vp, num_frames=NUM_FRAMES, resolution=RES)
            v = extract_velocity(score_fn, frames, args.device)
            np.save(out_path, v)
            written_index.append(rec)
            n_done += 1
        except Exception as e:
            log.warning("err %s: %s", vid, e)
            n_err += 1
        if (i + 1) % 50 == 0:
            log.info("[%d/%d] done=%d skip=%d err=%d", i + 1, len(records), n_done, n_skip, n_err)

    with labels_jsonl.open("w") as f:
        for rec in written_index:
            f.write(json.dumps(rec) + "\n")
    log.info("DONE done=%d skip=%d err=%d → index %s", n_done, n_skip, n_err, labels_jsonl)


if __name__ == "__main__":
    main()
