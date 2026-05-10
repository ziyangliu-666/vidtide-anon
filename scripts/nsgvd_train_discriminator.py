"""Retrain NSG-VD discriminator on cached VidTide velocity features.

Usage:
    # Stage 1: extract first
    python scripts/nsgvd_extract_features.py --split train
    python scripts/nsgvd_extract_features.py --split test

    # Stage 2: train + eval
    python scripts/nsgvd_train_discriminator.py --epochs 50

Outputs:
    Checkpoints/ft_nsgvd_best.pth   — best discriminator by test AUROC
    results/ft_nsgvd.json           — final metrics (AUROC, BACC, F1, n_real, n_fake)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nsgvd_train")

REPO = Path(__file__).resolve().parent.parent
NSGVD_DIR = REPO / "vendor" / "NSG-VD"
CACHE_ROOT = REPO / "data" / "ft_features" / "nsgvd"
CKPT_OUT = REPO / "Checkpoints" / "ft_nsgvd_best.pth"
RESULT_OUT = REPO / "results" / "ft_nsgvd.json"

SEED = 42
NUM_FRAMES = 8
RES = 224


def _import_nsgvd():
    if str(NSGVD_DIR) not in sys.path:
        sys.path.insert(0, str(NSGVD_DIR))
    cwd = os.getcwd()
    os.chdir(str(NSGVD_DIR))
    try:
        from models.deep_mmd import deep_MMD
        from models.tall import SingleSwinBlockDiscriminator
        from utils.mmd_utils import MMD_batch2
        return deep_MMD, SingleSwinBlockDiscriminator, MMD_batch2
    finally:
        os.chdir(cwd)


class CachedVelocityDataset(Dataset):
    """Loads (T, 3, RES, RES) fp32 velocity tensors from disk."""

    def __init__(self, split: str, label_filter: int | None = None):
        self.dir = CACHE_ROOT / split
        idx_path = CACHE_ROOT / f"{split}_index.jsonl"
        with idx_path.open() as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if label_filter is not None:
            recs = [r for r in recs if r["label"] == label_filter]
        # Keep only records whose .npy actually exists
        self.records = [r for r in recs if (self.dir / f"{r['video']}.npy").exists()]
        log.info("Dataset[%s, label=%s] N=%d", split, label_filter, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        v = np.load(self.dir / f"{rec['video']}.npy")  # (T, 3, RES, RES)
        return torch.from_numpy(v).float(), int(rec["label"])


def make_paired_iter(real_loader: DataLoader, fake_loader: DataLoader) -> Iterator:
    """Yield matched (real_batch, fake_batch) pairs each epoch.

    Uses the shorter loader's length so we never run out mid-batch (NSG-VD's
    train_dMMD breaks on size mismatch anyway). Does NOT cycle — each epoch
    is one full pass over the smaller pool.
    """
    n = min(len(real_loader), len(fake_loader))
    real_iter = iter(real_loader)
    fake_iter = iter(fake_loader)
    for _ in range(n):
        rb, _ = next(real_iter)
        fb, _ = next(fake_iter)
        if rb.shape[0] != fb.shape[0]:
            continue
        yield rb, fb


def train_one_epoch(model, real_loader, fake_loader, optimizer, device) -> dict:
    model.train()
    losses = []
    for rb, fb in make_paired_iter(real_loader, fake_loader):
        rb = rb.to(device, non_blocking=True)
        fb = fb.to(device, non_blocking=True)
        X = torch.cat([rb, fb], dim=0)
        optimizer.zero_grad()
        TEMP, ep, sigma, sigma0_u = model(X, rb.shape[0])
        mmd_value = -TEMP[0]
        mmd_std = torch.sqrt(TEMP[1] + 1e-8)
        STAT_u = mmd_value / mmd_std
        STAT_u.backward()
        optimizer.step()
        model.set_info(mmd_value.item(), ep, sigma, sigma0_u, STAT_u.item())
        losses.append(STAT_u.item())
    return {"stat_u_mean": float(np.mean(losses)) if losses else float("nan")}


@torch.no_grad()
def build_ref_features(model, real_train_loader, device, ref_n: int = 200):
    """Pull `ref_n` real samples from train pool to serve as MMD anchors."""
    chunks = []
    grabbed = 0
    for batch, _ in real_train_loader:
        chunks.append(batch)
        grabbed += batch.shape[0]
        if grabbed >= ref_n:
            break
    ref_data = torch.cat(chunks, dim=0)[:ref_n].to(device)
    _, feature_ref = model.net(ref_data, out_feature=True)
    log.info("Ref pool: N=%d feat_dim=%d", ref_data.shape[0], feature_ref.shape[1])
    return feature_ref.detach(), ref_data.detach()


@torch.no_grad()
def eval_test(model, test_loader, ref_features, ref_data, MMD_batch2, device) -> dict:
    model.eval()
    n_ref = ref_features.shape[0]
    ref_flat = ref_data.view(n_ref, -1)
    sigma = float(model.sigma.item())
    sigma0 = float(model.sigma0_u.item())
    ep = float(model.ep.item())

    scores: list[float] = []
    labels: list[int] = []
    for batch, lbl in test_loader:
        batch = batch.to(device, non_blocking=True)
        _, feat = model.net(batch, out_feature=True)
        for i in range(batch.shape[0]):
            v = batch[i:i + 1]
            f = feat[i:i + 1]
            Fea = torch.cat([ref_features, f], dim=0)
            Fea_org = torch.cat([ref_flat, v.view(1, -1)], dim=0)
            mmd2 = MMD_batch2(Fea, n_ref, Fea_org, sigma, sigma0, ep, is_smooth=True)
            scores.append(float(mmd2[0].item()))
            labels.append(int(lbl[i].item()))

    scores_a = np.array(scores)
    labels_a = np.array(labels)
    auroc = float(roc_auc_score(labels_a, scores_a))
    # Threshold at median for bACC/F1 (NSG-VD has no calibrated cutoff)
    thr = float(np.median(scores_a))
    pred = (scores_a > thr).astype(int)
    return {
        "auroc": auroc,
        "bacc": float(balanced_accuracy_score(labels_a, pred)),
        "f1": float(f1_score(labels_a, pred, zero_division=0)),
        "n_fake": int((labels_a == 1).sum()),
        "n_real": int((labels_a == 0).sum()),
        "thr_median": thr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ref-n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    deep_MMD, SingleSwinBlockDiscriminator, MMD_batch2 = _import_nsgvd()

    device = args.device if torch.cuda.is_available() else "cpu"
    discriminator = SingleSwinBlockDiscriminator(num_features=300)
    model = deep_MMD(
        discriminator=discriminator, sigma=1000, sigma0=0.1, epsilon=10,
        img_size=RES, is_yy_zero=True, is_smooth=True,
    ).to(device)

    real_train_ds = CachedVelocityDataset("train", label_filter=0)
    fake_train_ds = CachedVelocityDataset("train", label_filter=1)
    test_ds = CachedVelocityDataset("test")

    real_loader = DataLoader(real_train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True)
    fake_loader = DataLoader(fake_train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    ref_features, ref_data = build_ref_features(model, real_loader, device, ref_n=args.ref_n)
    best = {"auroc": -1.0}
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, real_loader, fake_loader, optimizer, device)
        log.info("ep %d  train_STAT_u=%.4e", epoch, tr["stat_u_mean"])
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Refresh ref features against the now-trained net
            ref_features, ref_data = build_ref_features(model, real_loader, device, ref_n=args.ref_n)
            metrics = eval_test(model, test_loader, ref_features, ref_data, MMD_batch2, device)
            log.info("ep %d  test AUROC=%.4f bACC=%.4f F1=%.4f", epoch,
                     metrics["auroc"], metrics["bacc"], metrics["f1"])
            if metrics["auroc"] > best["auroc"]:
                best = {**metrics, "epoch": epoch}
                CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
                # Save the trained MMD wrapper buffers + net weights
                model.set_info(0.0, float(model.ep.item()), float(model.sigma.item()),
                               float(model.sigma0_u.item()), 0.0)
                torch.save(model.state_dict(), CKPT_OUT)
                log.info("→ saved best ckpt: %s (auroc=%.4f)", CKPT_OUT, best["auroc"])

    RESULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"ft_nsgvd": {**best, "epochs_run": args.epochs, "lr": args.lr,
                        "batch_size": args.batch_size, "ref_n": args.ref_n}}
    RESULT_OUT.write_text(json.dumps(out, indent=2))
    log.info("DONE → %s", RESULT_OUT)


if __name__ == "__main__":
    main()
