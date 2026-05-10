"""Re-evaluate cached LP heads on bilibili-only test subset (no platform shortcut)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data/ft_features"
SPLIT_TEST = REPO / "data/splits/ft_demamba_test.jsonl"

# Build set of bili-only test video ids
bili_ids = set()
plat_by_id = {}
label_by_id = {}
for line in open(SPLIT_TEST):
    d = json.loads(line)
    plat_by_id[d["video"]] = d["platform"]
    label_by_id[d["video"]] = d["label"]
    if d["platform"] == "bilibili":
        bili_ids.add(d["video"])

def auroc(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]; P = (y==1).sum(); N = (y==0).sum()
    if P==0 or N==0: return float("nan")
    tp = np.cumsum(y==1); fp = np.cumsum(y==0)
    tpr = np.concatenate(([0.], tp/P)); fpr = np.concatenate(([0.], fp/N))
    return float(np.trapezoid(tpr, fpr))

print(f"bili test ids: {len(bili_ids)}")

results = {}
for bb in ["tsm", "swin", "i3d", "slowfast"]:
    X = np.load(FEAT / f"{bb}_test_X.npy")
    y = np.load(FEAT / f"{bb}_test_y.npy")
    ids = [l.strip() for l in open(FEAT / f"{bb}_test_ids.txt")]
    assert len(ids) == X.shape[0] == y.shape[0], (len(ids), X.shape, y.shape)

    head_state = torch.load(FEAT / f"{bb}_lp_fc1.pt", map_location="cpu", weights_only=True)
    feat_dim = X.shape[1]
    head = nn.Linear(feat_dim, 1)
    head.load_state_dict(head_state)
    head.eval()
    with torch.no_grad():
        s_all = torch.sigmoid(head(torch.from_numpy(X).float())).squeeze(-1).numpy()

    # Mask: bili-only
    mask_bili = np.array([i in bili_ids for i in ids])
    s_bili = s_all[mask_bili]; y_bili = y[mask_bili]
    n_real = int((y_bili==0).sum()); n_fake = int((y_bili==1).sum())

    # Mask: non-bili (cross-platform sanity)
    mask_xp = ~mask_bili
    s_xp = s_all[mask_xp]; y_xp = y[mask_xp]

    a_full = auroc(s_all, y)
    a_bili = auroc(s_bili, y_bili)
    a_xp   = auroc(s_xp, y_xp) if y_xp.sum() > 0 and (1-y_xp).sum() > 0 else float("nan")

    print(f"{bb:>10}  full={a_full:.4f}  bili-only={a_bili:.4f}  cross-plat={a_xp:.4f}  "
          f"(bili: {n_real} real, {n_fake} fake)")
    results[bb] = {"full": round(a_full,4), "bili_only": round(a_bili,4),
                   "cross_platform_only": round(a_xp,4) if not np.isnan(a_xp) else None,
                   "bili_n_real": n_real, "bili_n_fake": n_fake}

OUT = REPO / "results/ft_lp_bilionly.json"
OUT.write_text(json.dumps({"backbones": results,
                           "test_split": "ft_demamba_test.jsonl",
                           "bili_subset_size": len(bili_ids)}, indent=2))
print(f"\nwrote {OUT}")
