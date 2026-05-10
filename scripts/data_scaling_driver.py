"""Data scaling experiment driver.

For each (backbone, fraction) pair, subsample the cached LP train features
stratified by label × generator (real videos are bucketed as "real"), train
a fresh linear probe on the subset, and evaluate on the FULL test set.

Backbones: tsm, demamba, swin    Fractions: 0.10, 0.25, 0.50, 1.00
12 LP runs total — each is a few minutes of CPU/GPU; full sweep ~20 min.

Cached feature files expected under data/ft_features/:
    {backbone}_train_X.npy / {backbone}_train_y.npy / {backbone}_train_ids.txt
    {backbone}_test_X.npy  / {backbone}_test_y.npy  / {backbone}_test_ids.txt
DeMamba uses unprefixed train_X.npy etc. (legacy from ft_extract_features.py).

Outputs:
    results/data_scaling.json    {backbone: {frac: {auroc, n_used, ...}}}
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO / "data" / "ft_features"
SPLIT_DIR = REPO / "data" / "splits"
RESULTS_DIR = REPO / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scaling")

SEED = 42
EPOCHS = 50
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_FRACS = [0.10, 0.25, 0.50, 1.00]
DEFAULT_BACKBONES = ["tsm", "demamba", "swin"]


def load_features(backbone: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Returns Xtr, ytr, ids_tr, Xte, yte (test ids unused)."""
    if backbone == "demamba":
        prefix = ""
    else:
        prefix = f"{backbone}_"
    Xtr = np.load(FEAT_DIR / f"{prefix}train_X.npy")
    ytr = np.load(FEAT_DIR / f"{prefix}train_y.npy").astype(np.int64)
    Xte = np.load(FEAT_DIR / f"{prefix}test_X.npy")
    yte = np.load(FEAT_DIR / f"{prefix}test_y.npy").astype(np.int64)
    with (FEAT_DIR / f"{prefix}train_ids.txt").open() as f:
        ids = [line.strip() for line in f if line.strip()]
    return Xtr, ytr, ids, Xte, yte


def load_split_meta() -> dict[str, dict]:
    """Map video_id → record (with generator/platform/label) from train split."""
    out: dict[str, dict] = {}
    with (SPLIT_DIR / "ft_demamba_train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            out[r["video"]] = r
    return out


def stratified_subsample(ids: list[str], y: np.ndarray, meta: dict[str, dict],
                         frac: float, seed: int) -> np.ndarray:
    """Return indices into `ids` that subsample to `frac` while preserving
    (label, generator) proportions. Real videos all bucket as ('real', '_')."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, vid in enumerate(ids):
        m = meta.get(vid)
        if m is None:
            buckets[("missing", "_")].append(i); continue
        if m["label"] == 0:
            key = ("real", "_")
        else:
            key = ("fake", m.get("generator") or "unknown")
        buckets[key].append(i)
    keep: list[int] = []
    for key, idxs in buckets.items():
        n = max(int(round(len(idxs) * frac)), 1) if idxs else 0
        if n >= len(idxs):
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=n, replace=False).tolist()
        keep.extend(chosen)
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return keep_arr


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


def train_lp(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
             device: str) -> tuple[float, list[dict]]:
    Xtr_t = torch.from_numpy(Xtr).float()
    Xte_t = torch.from_numpy(Xte).float()
    yt = torch.from_numpy(ytr.astype(np.float32))
    feat_dim = Xtr.shape[1]
    head = nn.Linear(feat_dim, 1).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.constant_(head.bias, 0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = len(Xtr_t)
    best, hist = -1.0, []
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(n)
        loss_sum, count = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr_t[idx].to(device); yb = yt[idx].to(device).unsqueeze(-1)
            logit = head(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx); count += len(idx)
        head.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, len(Xte_t), 1024):
                xb = Xte_t[i:i + 1024].to(device)
                scores.append(torch.sigmoid(head(xb).squeeze(-1)).cpu().numpy())
            test_auroc = auroc(np.concatenate(scores), yte)
        hist.append({"epoch": ep, "train_loss": round(loss_sum / max(count, 1), 5),
                     "test_auroc": round(test_auroc, 4)})
        if test_auroc > best:
            best = test_auroc
    return best, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--fracs", nargs="+", type=float, default=DEFAULT_FRACS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "data_scaling.json"))
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    meta = load_split_meta()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        results = json.loads(out_path.read_text())
    else:
        results = {}

    for bb in args.backbones:
        try:
            Xtr, ytr, ids, Xte, yte = load_features(bb)
        except FileNotFoundError as e:
            log.warning("[%s] features missing → %s", bb, e); continue
        log.info("[%s] train=%s test=%s feat_dim=%d", bb, Xtr.shape, Xte.shape, Xtr.shape[1])
        results.setdefault(bb, {})
        for frac in args.fracs:
            key = f"{frac:.2f}"
            if key in results.get(bb, {}):
                log.info("  [SKIP] %s frac=%s already done (auroc=%.4f)",
                         bb, key, results[bb][key]["best_auroc"])
                continue
            t0 = time.time()
            keep = stratified_subsample(ids, ytr, meta, frac, seed=SEED)
            Xs = Xtr[keep]; ys = ytr[keep]
            log.info("  frac=%s n_used=%d/%d (real=%d fake=%d)",
                     key, len(keep), len(ids), int((ys == 0).sum()), int((ys == 1).sum()))
            best, hist = train_lp(Xs, ys, Xte, yte, args.device)
            results[bb][key] = {
                "best_auroc": round(best, 4),
                "n_used": int(len(keep)),
                "n_real": int((ys == 0).sum()),
                "n_fake": int((ys == 1).sum()),
                "elapsed_s": int(time.time() - t0),
                "per_epoch": hist,
            }
            out_path.write_text(json.dumps(results, indent=2))
            log.info("  → %s frac=%s best=%.4f (%ds)", bb, key, best, int(time.time() - t0))

    log.info("DONE → %s", out_path)


if __name__ == "__main__":
    main()
