"""Train a linear probe (binary AI-vs-real) on cached mmaction backbone features.

Loads data/ft_features/{backbone}_{train,test}_{X,y,ids}, trains a fresh
nn.Linear(feat_dim, 1) with BCE for N epochs, evals AUROC each epoch,
saves best head + result JSON.

Outputs:
  - results/ft_{backbone}_lp.json
  - data/ft_features/{backbone}_lp_fc1.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO / "data" / "ft_features"
RESULTS_DIR = REPO / "results"

SEED = 42
EPOCHS = 50
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


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


def eval_head(head, X, y, device):
    head.eval()
    with torch.no_grad():
        scores = []
        for i in range(0, len(X), 1024):
            xb = X[i:i+1024].to(device).float()
            scores.append(torch.sigmoid(head(xb).squeeze(-1)).cpu().numpy())
    return auroc(np.concatenate(scores), y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    args = ap.parse_args()
    backbone = args.backbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.from_numpy(np.load(FEAT_DIR / f"{backbone}_train_X.npy")).float()
    ytr = np.load(FEAT_DIR / f"{backbone}_train_y.npy").astype(np.int64)
    Xte = torch.from_numpy(np.load(FEAT_DIR / f"{backbone}_test_X.npy")).float()
    yte = np.load(FEAT_DIR / f"{backbone}_test_y.npy").astype(np.int64)
    print(f"{backbone} train: X={Xtr.shape} pos={int(ytr.sum())}")
    print(f"{backbone} test : X={Xte.shape} pos={int(yte.sum())}")
    feat_dim = Xtr.shape[1]

    torch.manual_seed(SEED); np.random.seed(SEED)
    head = nn.Linear(feat_dim, 1).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.constant_(head.bias, 0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    yt = torch.from_numpy(ytr).float()

    per_epoch = []
    best_auroc, best_state = 0.0, None
    n = len(Xtr)
    print(f"\n[LP TRAIN] {EPOCHS} epochs, batch={BATCH}, lr={LR}, wd={WEIGHT_DECAY}, dim={feat_dim}")
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(n)
        loss_sum, count = 0.0, 0
        t0 = time.time()
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = Xtr[idx].to(device)
            yb = yt[idx].to(device).unsqueeze(-1)
            logit = head(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx); count += len(idx)
        train_loss = loss_sum / count
        test_auroc = eval_head(head, Xte, yte, device)
        per_epoch.append({"epoch": ep, "train_loss": round(train_loss, 5), "test_auroc": round(test_auroc, 4)})
        flag = ""
        if test_auroc > best_auroc:
            best_auroc = test_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            flag = " *"
        print(f"  ep{ep:>3} loss={train_loss:.4f} test_auroc={test_auroc:.4f} ({time.time()-t0:.1f}s){flag}")

    torch.save(best_state, FEAT_DIR / f"{backbone}_lp_fc1.pt")
    out = {
        "backbone": backbone,
        "feat_dim": feat_dim,
        "best_lp_auroc": round(best_auroc, 4),
        "n_train": int(n), "n_test": int(len(Xte)), "test_pos": int(yte.sum()),
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "per_epoch": per_epoch,
    }
    out_path = RESULTS_DIR / f"ft_{backbone}_lp.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {out_path}\nbest LP AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
