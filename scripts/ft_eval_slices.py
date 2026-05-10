"""Per-platform and per-generator AUROC for both baseline Pika fc1 and the
fine-tuned LP head on the cached test features.

Compares each slice's AUROC under both heads; output highlights where the
gain is largest.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO / "data" / "ft_features"
RESULTS_DIR = REPO / "results"
DB = REPO / "data" / "vidtide.db"
CKPT = REPO / "vendor" / "NSG-VD" / "results" / "ckpts" / "baselines" / "standard-Pika-demamba" / "final_ckpt.pth"
FEAT_DIM = 197 * 768
MIN_SLICE = 30


def auroc(s, y):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    P = float((ys == 1).sum()); N = float((ys == 0).sum())
    if P == 0 or N == 0: return float("nan")
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    tpr = np.concatenate(([0.], tp / P))
    fpr = np.concatenate(([0.], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(np.load(FEAT_DIR / "test_X.npy")).float().to(device)
    y = np.load(FEAT_DIR / "test_y.npy").astype(np.int64)
    ids = (FEAT_DIR / "test_ids.txt").read_text().splitlines()

    # Baseline head
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    base = nn.Linear(FEAT_DIM, 1).to(device)
    base.weight.data = state["fc1.weight"].float().to(device)
    base.bias.data = state["fc1.bias"].float().to(device)
    base.eval()

    # FT head
    lp_state = torch.load(FEAT_DIR / "lp_fc1.pt", map_location=device)
    lp = nn.Linear(FEAT_DIM, 1).to(device)
    lp.load_state_dict(lp_state); lp.eval()

    with torch.no_grad():
        base_scores = torch.sigmoid(base(X).squeeze()).cpu().numpy()
        lp_scores = torch.sigmoid(lp(X).squeeze()).cpu().numpy()

    print(f"Overall — baseline AUROC = {auroc(base_scores, y):.4f}")
    print(f"Overall — LP       AUROC = {auroc(lp_scores, y):.4f}")
    print(f"Overall Δ = +{auroc(lp_scores, y) - auroc(base_scores, y):.4f}\n")

    # Join meta
    con = sqlite3.connect(DB)
    meta = {
        vid: (plat, gen or None)
        for vid, plat, gen in con.execute(
            "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator,''),'') FROM videos"
        )
    }
    con.close()

    # Per-platform: need both fakes and reals in slice; reals shared
    by_plat_fake_idx: dict = defaultdict(list)
    by_plat_real_idx: dict = defaultdict(list)
    for i, vid in enumerate(ids):
        m = meta.get(vid)
        if m is None: continue
        plat = m[0]
        if y[i] == 1: by_plat_fake_idx[plat].append(i)
        else: by_plat_real_idx[plat].append(i)

    # Per-generator: fakes only, reals shared (all reals)
    by_gen_idx: dict = defaultdict(list)
    real_idx_all = [i for i in range(len(y)) if y[i] == 0]
    for i, vid in enumerate(ids):
        if y[i] != 1: continue
        m = meta.get(vid)
        if m is None: continue
        gen = m[1] or "_unlabeled"
        by_gen_idx[gen].append(i)

    rows_plat = []
    print("=== Per-platform ===")
    print(f"{'platform':<10} {'n_fake':<7} {'n_real':<7} {'baseline':<10} {'LP':<10} {'Δ':<8}")
    for plat in sorted(by_plat_fake_idx.keys()):
        fake_idx = by_plat_fake_idx[plat]
        real_idx = by_plat_real_idx.get(plat) or real_idx_all
        if len(fake_idx) < MIN_SLICE: continue
        sel = np.array(fake_idx + real_idx)
        s_base = auroc(base_scores[sel], y[sel])
        s_lp = auroc(lp_scores[sel], y[sel])
        rows_plat.append({"platform": plat, "n_fake": len(fake_idx), "n_real": len(real_idx),
                          "baseline": round(s_base, 4), "lp": round(s_lp, 4), "delta": round(s_lp - s_base, 4),
                          "real_src": "same-plat" if by_plat_real_idx.get(plat) else "global"})
        print(f"{plat:<10} {len(fake_idx):<7} {len(real_idx):<7} {s_base:<10.4f} {s_lp:<10.4f} {s_lp-s_base:+.4f}")

    rows_gen = []
    print("\n=== Per-generator (all reals as negatives) ===")
    print(f"{'generator':<13} {'n_fake':<7} {'baseline':<10} {'LP':<10} {'Δ':<8}")
    for gen in sorted(by_gen_idx.keys(), key=lambda g: -len(by_gen_idx[g])):
        fake_idx = by_gen_idx[gen]
        if len(fake_idx) < MIN_SLICE: continue
        sel = np.array(fake_idx + real_idx_all)
        s_base = auroc(base_scores[sel], y[sel])
        s_lp = auroc(lp_scores[sel], y[sel])
        rows_gen.append({"generator": gen, "n_fake": len(fake_idx),
                         "baseline": round(s_base, 4), "lp": round(s_lp, 4), "delta": round(s_lp - s_base, 4)})
        print(f"{gen:<13} {len(fake_idx):<7} {s_base:<10.4f} {s_lp:<10.4f} {s_lp-s_base:+.4f}")

    out = {
        "overall": {"baseline": round(auroc(base_scores, y), 4),
                    "lp": round(auroc(lp_scores, y), 4),
                    "delta": round(auroc(lp_scores, y) - auroc(base_scores, y), 4)},
        "per_platform": rows_plat,
        "per_generator": rows_gen,
        "n_test": len(y), "n_test_pos": int(y.sum()),
    }
    out_path = RESULTS_DIR / "ft_demamba_lp_slices.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
