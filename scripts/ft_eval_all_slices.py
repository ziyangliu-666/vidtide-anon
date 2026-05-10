"""Per-generator and per-platform AUROC + F1 for all 5 LP backbones on M0 test.

Reads cached test features + trained LP heads, joins to vidtide.db for
generator/platform metadata, emits ``results/ft_all_lp_slices.json`` with
overall + per-slice (AUROC, F1@0.5) for DeMamba, VideoSwin, TSM, I3D, SlowFast.

Also includes an aggregate ``Other`` generator row for named generators with
fewer than MIN_SLICE fakes individually, and a Bilibili-controlled cross-tab.
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
MIN_SLICE = 10

BACKBONES = [
    ("demamba",  "test_X.npy",          "test_y.npy",          "test_ids.txt",          "lp_fc1.pt"),
    ("swin",     "swin_test_X.npy",     "swin_test_y.npy",     "swin_test_ids.txt",     "swin_lp_fc1.pt"),
    ("tsm",      "tsm_test_X.npy",      "tsm_test_y.npy",      "tsm_test_ids.txt",      "tsm_lp_fc1.pt"),
    ("i3d",      "i3d_test_X.npy",      "i3d_test_y.npy",      "i3d_test_ids.txt",      "i3d_lp_fc1.pt"),
    ("slowfast", "slowfast_test_X.npy", "slowfast_test_y.npy", "slowfast_test_ids.txt", "slowfast_lp_fc1.pt"),
]


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    P = float((ys == 1).sum()); N = float((ys == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    tpr = np.concatenate(([0.], tp / P))
    fpr = np.concatenate(([0.], fp / N))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def f1_at(s: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    pred = (s >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def tpr_at_fpr(scores_fake: np.ndarray, scores_real_all: np.ndarray, target_fpr: float = 0.05) -> float:
    """Recall on the slice's fakes at a fixed global FPR on the full real set.

    Threshold is chosen from the full real set so the operating point is
    slice-independent; then slice TPR is measured against that threshold.
    """
    if len(scores_fake) == 0:
        return float("nan")
    thr = np.quantile(scores_real_all, 1.0 - target_fpr)
    return float((scores_fake >= thr).mean())


def score(X_path: Path, head_path: Path, device: str) -> np.ndarray:
    X = torch.from_numpy(np.load(X_path)).float().to(device)
    feat_dim = X.shape[1]
    head = nn.Linear(feat_dim, 1).to(device)
    state = torch.load(head_path, map_location=device, weights_only=False)
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        s = torch.sigmoid(head(X).squeeze()).cpu().numpy()
    return s


def slice_metrics(scores_by_bb: dict[str, np.ndarray], y: np.ndarray, sel: np.ndarray,
                  fake_idx: np.ndarray, real_idx_all: np.ndarray) -> dict:
    """Compute AUROC on the full slice (fake+real) and TPR@5%FPR on the slice's
    fakes using the threshold derived from the *full* real set (global operating
    point, shared across slices for fair comparison).
    """
    row = {}
    for bb, s in scores_by_bb.items():
        row[bb] = {
            "auroc": round(auroc(s[sel], y[sel]), 4),
            "tpr5":  round(tpr_at_fpr(s[fake_idx], s[real_idx_all]), 4),
        }
    return row


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    con = sqlite3.connect(DB)
    meta = {
        vid: (plat, gen or None)
        for vid, plat, gen in con.execute(
            "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator,''),'') FROM videos"
        )
    }
    con.close()

    y_ref = np.load(FEAT_DIR / "test_y.npy").astype(np.int64)
    ids_ref = (FEAT_DIR / "test_ids.txt").read_text().splitlines()

    per_bb: dict[str, np.ndarray] = {}
    for bb, xn, yn, idn, hn in BACKBONES:
        y = np.load(FEAT_DIR / yn).astype(np.int64)
        ids = (FEAT_DIR / idn).read_text().splitlines()
        assert np.array_equal(y, y_ref) and ids == ids_ref, f"{bb}: label/id mismatch"
        per_bb[bb] = score(FEAT_DIR / xn, FEAT_DIR / hn, device)
        print(f"{bb:<10} overall AUROC={auroc(per_bb[bb], y_ref):.4f} F1={f1_at(per_bb[bb], y_ref):.4f}")

    # Indices
    real_idx_all = np.array([i for i in range(len(y_ref)) if y_ref[i] == 0])
    by_plat_fake: dict = defaultdict(list)
    by_plat_real: dict = defaultdict(list)
    by_gen_fake: dict = defaultdict(list)
    bili_by_gen_fake: dict = defaultdict(list)
    bili_real_idx: list = []
    for i, vid in enumerate(ids_ref):
        m = meta.get(vid)
        if m is None:
            continue
        plat, gen = m
        gen_key = gen or "_unlabeled"
        if y_ref[i] == 1:
            by_plat_fake[plat].append(i)
            by_gen_fake[gen_key].append(i)
            if plat == "bilibili":
                bili_by_gen_fake[gen_key].append(i)
        else:
            by_plat_real[plat].append(i)
            if plat == "bilibili":
                bili_real_idx.append(i)

    slices: list[dict] = []

    fake_idx_all = np.array([i for i in range(len(y_ref)) if y_ref[i] == 1])

    # Overall
    sel = np.arange(len(y_ref))
    slices.append({
        "group": "overall", "label": "All test videos",
        "n_fake": int((y_ref == 1).sum()), "n_real": int((y_ref == 0).sum()),
        "metrics": slice_metrics(per_bb, y_ref, sel, fake_idx_all, real_idx_all),
    })

    # By platform (shared reals fallback for platforms with no same-plat reals)
    for plat in ["bilibili", "reddit", "youtube"]:
        fake_idx = np.array(by_plat_fake.get(plat, []))
        real_idx = np.array(by_plat_real.get(plat) or list(real_idx_all))
        if len(fake_idx) < MIN_SLICE:
            continue
        sel = np.concatenate([fake_idx, real_idx])
        slices.append({
            "group": "platform", "label": plat.capitalize() if plat != "youtube" else "YouTube",
            "n_fake": len(fake_idx), "n_real": len(real_idx),
            "real_src": "same-plat" if by_plat_real.get(plat) else "global",
            "metrics": slice_metrics(per_bb, y_ref, sel, fake_idx, real_idx_all),
        })

    # By generator, ordered by n_fake desc
    gen_order = sorted(by_gen_fake.keys(), key=lambda g: -len(by_gen_fake[g]))
    other_idx: list = []
    other_members: list = []
    for gen in gen_order:
        fake_idx = np.array(by_gen_fake[gen])
        if len(fake_idx) < MIN_SLICE:
            other_idx.extend(fake_idx.tolist())
            other_members.append(f"{gen}:{len(fake_idx)}")
            continue
        sel = np.concatenate([fake_idx, real_idx_all])
        slices.append({
            "group": "generator", "label": gen,
            "n_fake": len(fake_idx), "n_real": len(real_idx_all),
            "metrics": slice_metrics(per_bb, y_ref, sel, fake_idx, real_idx_all),
        })
    if other_idx:
        other_arr = np.array(other_idx)
        sel = np.concatenate([other_arr, real_idx_all])
        slices.append({
            "group": "generator", "label": f"Other ({', '.join(other_members)})",
            "n_fake": len(other_idx), "n_real": len(real_idx_all),
            "metrics": slice_metrics(per_bb, y_ref, sel, other_arr, real_idx_all),
        })

    # Bilibili × labeled generators (platform-controlled)
    bili_real_arr = np.array(bili_real_idx)
    labeled_fake: list = []
    for gen in gen_order:
        if gen == "_unlabeled":
            continue
        labeled_fake.extend(bili_by_gen_fake.get(gen, []))
    if labeled_fake:
        labeled_arr = np.array(labeled_fake)
        sel = np.concatenate([labeled_arr, bili_real_arr])
        slices.append({
            "group": "crosstab", "label": "Bilibili $\\cap$ labeled gens",
            "n_fake": len(labeled_fake), "n_real": len(bili_real_arr),
            "metrics": slice_metrics(per_bb, y_ref, sel, labeled_arr, real_idx_all),
        })

    out = {
        "overall": {bb: {"auroc": round(auroc(per_bb[bb], y_ref), 4),
                         "tpr5": round(tpr_at_fpr(per_bb[bb][fake_idx_all], per_bb[bb][real_idx_all]), 4)}
                    for bb in per_bb},
        "slices": slices,
        "n_test": len(y_ref),
        "n_test_pos": int(y_ref.sum()),
        "min_slice": MIN_SLICE,
        "operating_point": "TPR at FPR=5% (global real-set threshold)",
    }
    out_path = RESULTS_DIR / "ft_all_lp_slices.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'group':<10} {'label':<22} {'n':<6}  " + "  ".join(f"{bb:>11}" for bb in per_bb))
    for s in slices:
        cells = "  ".join(
            f"{s['metrics'][bb]['auroc']:>5.3f}/{s['metrics'][bb]['tpr5']:>5.3f}"
            for bb in per_bb
        )
        print(f"{s['group']:<10} {s['label'][:22]:<22} {s['n_fake']:<6}  {cells}")

    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
