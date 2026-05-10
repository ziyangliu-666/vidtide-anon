"""Extract pre-fc1 DeMamba features for the FT train+test pool.

Loads `vendor/NSG-VD/.../standard-Pika-demamba/final_ckpt.pth`, monkey-patches
fc1 → Identity so model() returns the 197*768=151,296-dim video feature.
For each video in data/splits/ft_demamba_{train,test}.jsonl, extracts 8 frames,
runs encoder+mamba forward, dumps feature as a row in two big .npy files plus
parallel labels/ids files.

Outputs (under data/ft_features/):
  - train_X.npy  (N, 151296) float16
  - train_y.npy  (N,) uint8
  - train_ids.txt  (N lines)
  - test_X.npy / test_y.npy / test_ids.txt
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
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.detection.dataset import extract_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ft_extract")

NSGVD_DIR = REPO / "vendor" / "NSG-VD"
CKPT = NSGVD_DIR / "results" / "ckpts" / "baselines" / "standard-Pika-demamba" / "final_ckpt.pth"
SPLIT_DIR = REPO / "data" / "splits"
BLOB_ROOT = REPO / "data" / "blobs" / "videos"
OUT_DIR = REPO / "data" / "ft_features"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
FEAT_DIM = 197 * 768  # 151,296


def load_split(name: str) -> list[dict]:
    rows = []
    with (SPLIT_DIR / f"ft_demamba_{name}.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def video_path(vid: str, label: int) -> Path:
    sub = "fake" if label == 1 else "real"
    return BLOB_ROOT / sub / f"{vid}.mp4"


def build_model(device: str) -> nn.Module:
    if str(NSGVD_DIR) not in sys.path:
        sys.path.insert(0, str(NSGVD_DIR))
    from models.demamba import XCLIP_DeMamba
    m = XCLIP_DeMamba()
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    m.load_state_dict(state)
    m.fc1 = nn.Identity()
    m.dropout = nn.Identity()
    m.to(device).eval()
    return m


def extract_one(sample: tuple[str, int], num_frames: int, resolution: int):
    vid, label = sample
    p = video_path(vid, label)
    try:
        frames = extract_frames(p, num_frames=num_frames, resolution=resolution)
        return vid, label, frames, None
    except Exception as e:
        return vid, label, None, str(e)


def iter_samples(samples, num_frames, resolution, prefetch):
    if prefetch <= 0:
        for s in samples:
            yield extract_one(s, num_frames, resolution)
        return
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        yield from pool.map(lambda s: extract_one(s, num_frames, resolution), samples)


def run_split(name: str, model: nn.Module, device: str, mean, std, prefetch: int) -> None:
    rows = load_split(name)
    log.info("%s: %d videos", name, len(rows))
    samples = [(r["video"], r["label"]) for r in rows]

    feats = np.zeros((len(samples), FEAT_DIM), dtype=np.float16)
    labels = np.zeros(len(samples), dtype=np.uint8)
    ids = [""] * len(samples)
    valid = np.zeros(len(samples), dtype=bool)

    t0 = time.time()
    errs = 0
    written = 0
    for i, (vid, label, frames, err) in enumerate(iter_samples(samples, 8, 224, prefetch)):
        if err is not None:
            log.warning("frame err %s: %s", vid, err)
            errs += 1
            continue
        try:
            x = torch.from_numpy(frames).float().to(device) / 255.0
            x = x.permute(0, 3, 1, 2).unsqueeze(0)
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                B, T = x.shape[0], x.shape[1]
                x = F.interpolate(
                    x.view(B * T, 3, x.shape[-2], x.shape[-1]),
                    size=(224, 224), mode="bilinear", align_corners=False,
                ).view(B, T, 3, 224, 224)
            x = (x - mean) / std
            with torch.no_grad():
                feat = model(x).cpu().numpy().astype(np.float16).squeeze()
            feats[written] = feat
            labels[written] = label
            ids[written] = vid
            valid[written] = True
            written += 1
        except Exception as e:
            log.warning("model err %s: %s", vid, e)
            errs += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(samples) - i - 1) / max(rate, 0.001)
            log.info("  [%d/%d] elapsed=%ds rate=%.2fv/s eta=%dm errs=%d",
                     i + 1, len(samples), int(elapsed), rate, int(eta / 60), errs)

    # Trim
    feats = feats[:written]
    labels = labels[:written]
    ids = ids[:written]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}_X.npy", feats)
    np.save(OUT_DIR / f"{name}_y.npy", labels)
    with (OUT_DIR / f"{name}_ids.txt").open("w") as f:
        for v in ids: f.write(v + "\n")
    log.info("%s: wrote %d feats (%d errors), elapsed=%ds → %s",
             name, written, errs, int(time.time() - t0), OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "train"],
                    help="Which splits to extract (test first by default — smaller).")
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)
    model = build_model(device)
    log.info("model loaded; fc1=Identity, params trainable: %d",
             sum(p.numel() for p in model.parameters() if p.requires_grad))

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    for s in args.splits:
        run_split(s, model, device, mean, std, args.prefetch)


if __name__ == "__main__":
    main()
