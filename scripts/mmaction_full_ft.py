"""Full fine-tune of a K400-pretrained mmaction2 backbone on the VidTide
ft_demamba_{train,test}.jsonl split (8K/2K, binary AI vs real).

Differs from `mmaction_train_lp.py` in three ways:
  1. ALL backbone params are unfrozen — not just the linear head.
  2. End-to-end frame loading per epoch (no cached features) → enables augmentation.
  3. Mixed-precision (bf16) + AdamW(cos→1e-6) + early stop on val AUROC.

Usage:
    python scripts/mmaction_full_ft.py --backbone tsm
    python scripts/mmaction_full_ft.py --backbone swin --epochs 20 --batch-size 8

Outputs:
    results/ft_{backbone}_full.json
    Checkpoints/ft_{backbone}_full.pth   (best by val AUROC)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.detection.dataset import extract_frames  # noqa: E402
from scripts.mmaction_extract_features import (  # noqa: E402
    BACKBONES, IMAGENET_MEAN, IMAGENET_STD, build_model, load_split, video_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mmft")

CKPT_OUT = REPO / "Checkpoints"
RESULTS_DIR = REPO / "results"
SEED = 42


# ───────────────────────── data ─────────────────────────────────────────────

class FFTDataset(Dataset):
    """Loads num_frames RGB frames per video and (optionally) augments."""

    def __init__(self, split: str, num_frames: int, resolution: int = 224, train: bool = True):
        self.records = load_split(split)
        self.num_frames = num_frames
        self.resolution = resolution
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            frames = extract_frames(video_path(rec["video"], rec["label"]),
                                    num_frames=self.num_frames, resolution=self.resolution)
        except Exception:
            return None  # collate filters None
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        if self.train:
            x = self._augment(x)
        return x, int(rec["label"])

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # HFlip same across all frames
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        # ColorJitter strength 0.1 — same params for whole clip
        b = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        x = (x * b).clamp(0, 1)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = ((x - mean) * c + mean).clamp(0, 1)
        # RandomResizedCrop scale 0.8-1.0, same crop for whole clip
        T, _, H, W = x.shape
        s = float(torch.empty(1).uniform_(0.8, 1.0).item())
        nh = max(int(round(H * math.sqrt(s))), 8)
        nw = max(int(round(W * math.sqrt(s))), 8)
        top = int(torch.randint(0, H - nh + 1, (1,)).item())
        left = int(torch.randint(0, W - nw + 1, (1,)).item())
        x = x[:, :, top:top + nh, left:left + nw]
        x = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        return x


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch], dim=0)  # (B, T, 3, H, W)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return xs, ys


# ───────────────────────── model wrapper ────────────────────────────────────

class FTWrapper(nn.Module):
    """Backbone + adaptive pool + binary linear head. Mirrors featurize() pooling."""

    def __init__(self, backbone, spec: dict, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.head = nn.Linear(feat_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def pool(self, feat) -> torch.Tensor:
        if isinstance(feat, (tuple, list)) and len(feat) == 2 and all(t.dim() == 5 for t in feat):
            pooled = [F.adaptive_avg_pool3d(f, 1).flatten(1) for f in feat]
            return torch.cat(pooled, dim=1)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.dim() == 5:
            return F.adaptive_avg_pool3d(feat, 1).flatten(1)
        if feat.dim() == 4:
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat.view(feat.shape[0], -1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B, T, 3, H, W) in [0, 1]
        B, T = x_raw.shape[0], x_raw.shape[1]
        x = x_raw * 255.0
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        x = (x - mean) / std
        if self.spec["is_3d"]:
            inp = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)
            feat = self.backbone(inp)
            v = self.pool(feat)
        else:
            inp = x.reshape(B * T, 3, x.shape[-2], x.shape[-1])
            feat = self.backbone(inp)
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]
            if feat.dim() == 4:
                p = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B*T, C)
                v = p.view(B, T, -1).mean(dim=1)
            else:
                v = self.pool(feat)
        return self.head(v).squeeze(-1)  # (B,)


def probe_feat_dim(backbone_name: str, model, spec: dict, device: str) -> int:
    T = spec["num_frames"]
    dummy = torch.zeros(1, T, 3, 224, 224, device=device)
    wrapper = FTWrapper(model.backbone, spec, feat_dim=1)  # feat_dim placeholder
    wrapper.to(device).eval()
    with torch.no_grad():
        x = dummy * 255.0
        mean = IMAGENET_MEAN.to(device); std = IMAGENET_STD.to(device)
        x = (x - mean) / std
        if spec["is_3d"]:
            inp = x.permute(0, 2, 1, 3, 4).contiguous()
            feat = wrapper.backbone(inp)
            v = wrapper.pool(feat)
        else:
            B = 1
            inp = x.reshape(B * T, 3, 224, 224)
            feat = wrapper.backbone(inp)
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]
            p = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            v = p.view(B, T, -1).mean(dim=1)
    return int(v.shape[1])


# ───────────────────────── train / eval ─────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    P = float((y == 1).sum()); N = float((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    tpr = np.concatenate(([0.], tp / P))
    fpr = np.concatenate(([0.], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


@torch.no_grad()
def evaluate(model: FTWrapper, loader: DataLoader, device: str) -> tuple[float, int]:
    model.eval()
    scores: list[float] = []; labels: list[int] = []; n_err = 0
    for batch in loader:
        if batch is None:
            n_err += 1; continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logit = model(x)
        scores.extend(torch.sigmoid(logit).float().cpu().numpy().tolist())
        labels.extend(y.long().tolist())
    return auroc(np.asarray(scores), np.asarray(labels)), n_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES.keys()))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-frames", type=int, default=None,
                    help="Override spec num_frames (default: backbone-specific)")
    ap.add_argument("--resolution", type=int, default=224)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = args.device if torch.cuda.is_available() else "cpu"

    log.info("device=%s backbone=%s epochs=%d bs=%d lr=%g",
             device, args.backbone, args.epochs, args.batch_size, args.lr)
    base_model, spec = build_model(args.backbone, device)
    if args.num_frames is not None:
        spec = dict(spec); spec["num_frames"] = args.num_frames
    feat_dim = probe_feat_dim(args.backbone, base_model, spec, device)
    log.info("feat_dim=%d num_frames=%d", feat_dim, spec["num_frames"])

    model = FTWrapper(base_model.backbone, spec, feat_dim=feat_dim).to(device)
    model.train()

    train_ds = FFTDataset("train", num_frames=spec["num_frames"],
                          resolution=args.resolution, train=True)
    test_ds = FFTDataset("test", num_frames=spec["num_frames"],
                         resolution=args.resolution, train=False)
    log.info("train=%d test=%d", len(train_ds), len(test_ds))
    loader_kw = {}
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
        loader_kw["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate, **loader_kw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_ep = max(len(train_loader), 1)
    total_steps = steps_per_ep * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = None  # bf16 needs no scaler

    per_epoch: list[dict] = []
    best_auroc = -1.0
    best_epoch = 0
    no_improve = 0
    CKPT_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_OUT / f"ft_{args.backbone}_full.pth"
    out_path = RESULTS_DIR / f"ft_{args.backbone}_full.json"

    n_batches = len(train_loader)
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_n, n_err = 0.0, 0, 0
        t0 = time.time()
        t_step = time.time()
        for step, batch in enumerate(train_loader, 1):
            if batch is None:
                n_err += 1; continue
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = model(x)
                loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item() * x.shape[0]; ep_n += x.shape[0]
            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t_step
                log.info("ep %d  step %d/%d  loss=%.4f  %.2fs/step",
                         ep, step, n_batches, loss.item(), dt / max(args.log_every if step > 1 else 1, 1))
                t_step = time.time()
        train_loss = ep_loss / max(ep_n, 1)
        val_auroc, val_err = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        flag = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = ep
            no_improve = 0
            torch.save({"head": model.head.state_dict(),
                        "backbone": model.backbone.state_dict(),
                        "spec": spec,
                        "best_auroc": best_auroc,
                        "epoch": ep}, ckpt_path)
            flag = " *"
        else:
            no_improve += 1
        per_epoch.append({"epoch": ep, "train_loss": round(train_loss, 5),
                          "val_auroc": round(val_auroc, 4), "elapsed_s": int(elapsed),
                          "lr": opt.param_groups[0]["lr"]})
        log.info("ep %d  loss=%.4f  val_AUROC=%.4f  (%ds, errs=%d/%d)%s",
                 ep, train_loss, val_auroc, int(elapsed), n_err, val_err, flag)
        if no_improve >= args.patience:
            log.info("early stop @ ep %d (no improve for %d)", ep, args.patience)
            break

    out = {
        f"ft_{args.backbone}_full": {
            "backbone": args.backbone,
            "feat_dim": feat_dim,
            "best_val_auroc": round(best_auroc, 4),
            "best_epoch": best_epoch,
            "epochs_run": len(per_epoch),
            "epochs_max": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "wd": args.weight_decay,
            "num_frames": spec["num_frames"], "resolution": args.resolution,
            "n_train": len(train_ds), "n_test": len(test_ds),
            "ckpt": str(ckpt_path),
            "per_epoch": per_epoch,
        }
    }
    out_path.write_text(json.dumps(out, indent=2))
    log.info("DONE → %s  best=%.4f @ ep %d", out_path, best_auroc, best_epoch)


if __name__ == "__main__":
    main()
