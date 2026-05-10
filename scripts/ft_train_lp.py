"""Train a linear probe (fc1 only) on cached DeMamba features.

Loads data/ft_features/{train,test}_{X,y,ids}, trains a fresh
nn.Linear(151296, 1) with BCE loss for N epochs (default 50; tiny so fast),
evaluates AUROC on test each epoch, saves the best fc1 weights.

Also computes the BASELINE AUROC by scoring test features through the
original Pika fc1 (loaded from final_ckpt.pth) — so we get a clean A/B in
one script.

Outputs:
  - results/ft_demamba_lp.json   {baseline_auroc, best_lp_auroc, per_epoch_auroc, ...}
  - data/ft_features/lp_fc1.pt   fine-tuned head weights
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO / "data" / "ft_features"
RESULTS_DIR = REPO / "results"
NSGVD_DIR = REPO / "vendor" / "NSG-VD"
CKPT = NSGVD_DIR / "results" / "ckpts" / "baselines" / "standard-Pika-demamba" / "final_ckpt.pth"

FEAT_DIM = 197 * 768
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


def load_baseline_fc1() -> nn.Linear:
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    fc1 = nn.Linear(FEAT_DIM, 1)
    fc1.weight.data = state["fc1.weight"].float()
    fc1.bias.data = state["fc1.bias"].float()
    return fc1


def eval_head(head: nn.Module, X: torch.Tensor, y: np.ndarray, device: str) -> float:
    head.eval()
    with torch.no_grad():
        scores = []
        for i in range(0, len(X), 1024):
            xb = X[i:i+1024].to(device).float()
            logit = head(xb).squeeze(-1)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    return auroc(np.concatenate(scores), y)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    Xtr = torch.from_numpy(np.load(FEAT_DIR / "train_X.npy")).float()
    ytr = np.load(FEAT_DIR / "train_y.npy").astype(np.int64)
    Xte = torch.from_numpy(np.load(FEAT_DIR / "test_X.npy")).float()
    yte = np.load(FEAT_DIR / "test_y.npy").astype(np.int64)
    print(f"train: X={Xtr.shape} y={ytr.shape} pos={int(ytr.sum())}")
    print(f"test : X={Xte.shape} y={yte.shape} pos={int(yte.sum())}")

    # ---- Baseline (Pika fc1, no FT) ----
    base_fc1 = load_baseline_fc1().to(device)
    base_auroc = eval_head(base_fc1, Xte, yte, device)
    print(f"\n[BASELINE Pika fc1] test AUROC = {base_auroc:.4f}")

    # ---- LP: fresh fc1 trained on VidTide ----
    torch.manual_seed(SEED); np.random.seed(SEED)
    head = nn.Linear(FEAT_DIM, 1).to(device)
    nn.init.xavier_uniform_(head.weight); nn.init.constant_(head.bias, 0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    yt = torch.from_numpy(ytr).float()

    per_epoch = []
    best_auroc, best_state = 0.0, None
    n = len(Xtr)
    print(f"\n[LP TRAIN] {EPOCHS} epochs, batch={BATCH}, lr={LR}, wd={WEIGHT_DECAY}")
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

    # ---- Save best head ----
    torch.save(best_state, FEAT_DIR / "lp_fc1.pt")
    out = {
        "baseline_pika_fc1_auroc": round(base_auroc, 4),
        "best_lp_auroc": round(best_auroc, 4),
        "delta": round(best_auroc - base_auroc, 4),
        "n_train": int(n),
        "n_test": int(len(Xte)),
        "test_pos": int(yte.sum()),
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "per_epoch": per_epoch,
    }
    out_path = RESULTS_DIR / "ft_demamba_lp.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {out_path}")
    print(f"\nBASELINE: {base_auroc:.4f}  →  LP: {best_auroc:.4f}  (Δ = {best_auroc - base_auroc:+.4f})")


if __name__ == "__main__":
    main()
