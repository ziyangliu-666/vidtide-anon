"""Quick eval of best NSG-VD discriminator ckpt on a 100R+100F balanced subset."""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "vendor" / "NSG-VD"))
os.chdir(REPO / "vendor" / "NSG-VD")
from nsgvd_train_discriminator import CachedVelocityDataset, _import_nsgvd, RES, build_ref_features  # noqa
os.chdir(REPO)

CKPT = REPO / "Checkpoints" / "ft_nsgvd_best.pth"
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--n", type=int, default=100); _ap.add_argument("--full", action="store_true")
_args = _ap.parse_args()
N_PER_CLASS = _args.n
SEED = 42

deep_MMD, SingleSwinBlockDiscriminator, MMD_batch2 = _import_nsgvd()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

discriminator = SingleSwinBlockDiscriminator(num_features=300)
model = deep_MMD(discriminator=discriminator, sigma=1000, sigma0=0.1, epsilon=10,
                 img_size=RES, is_yy_zero=True, is_smooth=True).to(device)
state = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(state)
model.eval()
print(f"Loaded {CKPT}")

real_train = CachedVelocityDataset("train", label_filter=0)
real_loader = DataLoader(real_train, batch_size=8, shuffle=True, num_workers=2)
ref_features, ref_data = build_ref_features(model, real_loader, device, ref_n=200)

real_test = CachedVelocityDataset("test", label_filter=0)
fake_test = CachedVelocityDataset("test", label_filter=1)
print(f"Test pool: {len(real_test)} real, {len(fake_test)} fake")

if _args.full:
    real_idx = np.arange(len(real_test))
    fake_idx = np.arange(len(fake_test))
    print(f"Using FULL test set: {len(real_idx)} real, {len(fake_idx)} fake")
else:
    rng = np.random.default_rng(SEED)
    real_idx = rng.choice(len(real_test), N_PER_CLASS, replace=False)
    fake_idx = rng.choice(len(fake_test), N_PER_CLASS, replace=False)

n_ref = ref_features.shape[0]
ref_flat = ref_data.view(n_ref, -1)
sigma = float(model.sigma.item()); sigma0 = float(model.sigma0_u.item()); ep = float(model.ep.item())
print(f"sigma={sigma:.3g} sigma0={sigma0:.3g} ep={ep:.3g}")

def score(ds, indices):
    out = []
    with torch.no_grad():
        for i in indices:
            v, lbl = ds[int(i)]
            v = v.unsqueeze(0).to(device)
            _, f = model.net(v, out_feature=True)
            Fea = torch.cat([ref_features, f], dim=0)
            Fea_org = torch.cat([ref_flat, v.view(1, -1)], dim=0)
            mmd2 = MMD_batch2(Fea, n_ref, Fea_org, sigma, sigma0, ep, is_smooth=True)
            out.append(float(mmd2[0].item()))
    return np.array(out)

real_scores = score(real_test, real_idx)
fake_scores = score(fake_test, fake_idx)
print(f"\nReal MMD scores: mean={real_scores.mean():.4g} std={real_scores.std():.4g} min={real_scores.min():.4g} max={real_scores.max():.4g}")
print(f"Fake MMD scores: mean={fake_scores.mean():.4g} std={fake_scores.std():.4g} min={fake_scores.min():.4g} max={fake_scores.max():.4g}")
print(f"Score separation (fake.mean - real.mean) / pooled.std = {(fake_scores.mean()-real_scores.mean())/np.sqrt((real_scores.var()+fake_scores.var())/2):.3f}")

scores = np.concatenate([real_scores, fake_scores])
labels = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
auroc = roc_auc_score(labels, scores)
thr = float(np.median(scores))
pred = (scores > thr).astype(int)
bacc = balanced_accuracy_score(labels, pred)
f1 = f1_score(labels, pred, zero_division=0)

print(f"\n{'='*50}")
print(f"100R+100F EVAL (ep 10 best ckpt):")
print(f"  AUROC = {auroc:.4f}")
print(f"  bACC  = {bacc:.4f}")
print(f"  F1    = {f1:.4f}")
print(f"  thr_median = {thr:.4g}")
print(f"{'='*50}")

# Also report sign-flipped AUROC (in case scoring direction is inverted)
auroc_inv = roc_auc_score(labels, -scores)
print(f"  AUROC (sign-flipped) = {auroc_inv:.4f}  ← higher MMD means more 'real' if this is bigger")

print("\n--- score histograms (10 bins) ---")
for name, s in [("real", real_scores), ("fake", fake_scores)]:
    h, edges = np.histogram(s, bins=10, range=(min(real_scores.min(), fake_scores.min()), max(real_scores.max(), fake_scores.max())))
    print(f"{name}: ", end="")
    for c in h: print(f"{c:3d}", end=" ")
    print()
