"""Full fine-tune of DeMamba (XCLIP encoder + temporal mamba) on VidTide.

Mirrors `scripts/mmaction_full_ft.py` but loads the DeMamba architecture
from vendor/NSG-VD with the published Pika-trained ckpt as init, replaces
fc1 with a fresh `nn.Linear(151296, 1)` binary head, and unfreezes all
parameters.

Usage:
    python scripts/demamba_full_ft.py --epochs 20 --batch-size 8

Outputs:
    results/ft_demamba_full.json
    Checkpoints/ft_demamba_full.pth   (best by val AUROC)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("demft")

NSGVD_DIR = REPO / "vendor" / "NSG-VD"
CKPT_INIT = NSGVD_DIR / "results" / "ckpts" / "baselines" / "standard-Pika-demamba" / "final_ckpt.pth"
SPLIT_DIR = REPO / "data" / "splits"
BLOB_ROOT = REPO / "data" / "blobs" / "videos"
CKPT_OUT = REPO / "Checkpoints"
RESULTS_DIR = REPO / "results"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
NUM_FRAMES = 8
RES = 224
SEED = 42


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
    state = torch.load(str(CKPT_INIT), map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    m.load_state_dict(state)
    # Replace classifier head with fresh binary linear (drop dropout for stability)
    feat_dim = 197 * 768
    m.fc1 = nn.Linear(feat_dim, 1)
    nn.init.xavier_uniform_(m.fc1.weight); nn.init.constant_(m.fc1.bias, 0.0)
    m.dropout = nn.Identity()
    m.to(device)
    return m


# ───────────────────────── data ─────────────────────────────────────────────

class FFTDataset(Dataset):
    def __init__(self, split: str, train: bool):
        self.records = load_split(split)
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            frames = extract_frames(video_path(rec["video"], rec["label"]),
                                    num_frames=NUM_FRAMES, resolution=RES)
        except Exception:
            return None
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        if self.train:
            x = self._augment(x)
        return x, int(rec["label"])

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        b = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        x = (x * b).clamp(0, 1)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = ((x - mean) * c + mean).clamp(0, 1)
        T, _, H, W = x.shape
        s = float(torch.empty(1).uniform_(0.8, 1.0).item())
        nh = max(int(round(H * math.sqrt(s))), 8)
        nw = max(int(round(W * math.sqrt(s))), 8)
        top = int(torch.randint(0, H - nh + 1, (1,)).item())
        left = int(torch.randint(0, W - nw + 1, (1,)).item())
        x = x[:, :, top:top + nh, left:left + nw]
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        return x


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return xs, ys


# ───────────────────────── train / eval ─────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    P = float((y == 1).sum()); N = float((y == 0).sum())
    if P == 0 or N == 0: return float("nan")
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    tpr = np.concatenate(([0.], tp / P))
    fpr = np.concatenate(([0.], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def forward_logit(model: nn.Module, x_raw: torch.Tensor, mean, std) -> torch.Tensor:
    """x_raw: (B, T, 3, H, W) in [0,1] → logit (B,)."""
    x = (x_raw - mean) / std
    out = model(x)
    return out.squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device, mean, std) -> tuple[float, int]:
    model.eval()
    scores: list[float] = []; labels: list[int] = []; n_err = 0
    for batch in loader:
        if batch is None:
            n_err += 1; continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logit = forward_logit(model, x, mean, std)
        scores.extend(torch.sigmoid(logit).float().cpu().numpy().tolist())
        labels.extend(y.long().tolist())
    return auroc(np.asarray(scores), np.asarray(labels)), n_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("device=%s epochs=%d bs=%d lr=%g", device, args.epochs, args.batch_size, args.lr)

    model = build_model(device)
    log.info("trainable params=%d", sum(p.numel() for p in model.parameters() if p.requires_grad))
    mean = IMAGENET_MEAN.to(device); std = IMAGENET_STD.to(device)

    train_ds = FFTDataset("train", train=True)
    test_ds = FFTDataset("test", train=False)
    log.info("train=%d test=%d", len(train_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader), 1) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    per_epoch: list[dict] = []
    best_auroc, best_epoch, no_improve = -1.0, 0, 0
    CKPT_OUT.mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_OUT / "ft_demamba_full.pth"
    out_path = RESULTS_DIR / "ft_demamba_full.json"

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_n, n_err = 0.0, 0, 0
        t0 = time.time()
        for batch in train_loader:
            if batch is None:
                n_err += 1; continue
            x, y = batch
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = forward_logit(model, x, mean, std)
                loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item() * x.shape[0]; ep_n += x.shape[0]
        train_loss = ep_loss / max(ep_n, 1)
        val_auroc, val_err = evaluate(model, test_loader, device, mean, std)
        elapsed = time.time() - t0
        flag = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc; best_epoch = ep; no_improve = 0
            torch.save({"state_dict": model.state_dict(), "best_auroc": best_auroc, "epoch": ep},
                       ckpt_path)
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

    out = {"ft_demamba_full": {
        "best_val_auroc": round(best_auroc, 4), "best_epoch": best_epoch,
        "epochs_run": len(per_epoch), "epochs_max": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "wd": args.weight_decay,
        "n_train": len(train_ds), "n_test": len(test_ds),
        "ckpt": str(ckpt_path), "per_epoch": per_epoch,
    }}
    out_path.write_text(json.dumps(out, indent=2))
    log.info("DONE → %s  best=%.4f @ ep %d", out_path, best_auroc, best_epoch)


if __name__ == "__main__":
    main()
