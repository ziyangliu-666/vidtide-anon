"""E1b: does refreshing training coverage recover accuracy on later releases?
V0 = M0 train; V1 = M0 train + n M0 clips in the new-slice preprocessing (control);
V2 = M0 train + n clips from M1-M4 (same preprocessing). Test on M5-M7. Same LP recipe for all
(AdamW lr 1e-3, wd 1e-4, batch 256, xavier init), fixed epochs, 3 seeds."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
R = str(__import__('pathlib').Path(__file__).resolve().parents[3]) + '/data/'
Xm0 = torch.from_numpy(np.load(R + 'ft_features/train_X.npy')); ym0 = torch.from_numpy(np.load(R + 'ft_features/train_y.npy').astype(np.float32))
def load(name):
    X = np.load(R + f'xfeat/{name}_X.npy'); M = [json.loads(l) for l in open(R + f'xfeat/{name}_meta.jsonl')]
    return torch.from_numpy(X), M
Xr, Mr = load('m0ref'); Xn, Mn = load('newmonthly')
yr = torch.tensor([m['label'] for m in Mr], dtype=torch.float32); yn = torch.tensor([m['label'] for m in Mn], dtype=torch.float32)
gn = np.array([m['group'] for m in Mn])
tr_new = np.isin(gn, ['2026-05', '2026-06', '2026-07', '2026-08']); te = np.isin(gn, ['2026-09', '2026-10', '2026-11'])
n_add = int(tr_new.sum())
dev = 'cuda'; EPOCHS = 30
def train(X, y, seed):
    torch.manual_seed(seed)
    h = nn.Linear(X.shape[1], 1).to(dev); nn.init.xavier_uniform_(h.weight); nn.init.constant_(h.bias, 0)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = F.binary_cross_entropy_with_logits(h(X[idx].to(dev).float()).squeeze(-1), y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return h
def score(h, X):
    with torch.no_grad():
        return torch.cat([h(X[i:i + 512].to(dev).float()).squeeze(-1).cpu() for i in range(0, len(X), 512)]).numpy()
Xte, yte = Xn[te], yn[te].numpy()
res = {'n_add': n_add, 'n_test': int(te.sum()), 'variants': {}}
rng = np.random.default_rng(0)
ctrl_idx = rng.choice(len(Xr), size=min(n_add, len(Xr)), replace=False)
variants = {
    'V0_M0': (Xm0, ym0),
    'V1_M0+M0windows(control)': (torch.cat([Xm0, Xr[ctrl_idx]]), torch.cat([ym0, yr[ctrl_idx]])),
    'V2_M0+M1-M4': (torch.cat([Xm0, Xn[tr_new]]), torch.cat([ym0, yn[tr_new]])),
}
for name, (X, y) in variants.items():
    aucs, per_month = [], {}
    for seed in (0, 1, 2):
        s = score(train(X, y, seed), Xte); aucs.append(roc_auc_score(yte, s))
        for mth in ['2026-09', '2026-10', '2026-11']:
            mk = gn[te] == mth
            per_month.setdefault(mth, []).append(roc_auc_score(yte[mk], s[mk]))
    res['variants'][name] = dict(auroc_mean=float(np.mean(aucs)), auroc_seeds=[float(a) for a in aucs],
                                 per_month={k: float(np.mean(v)) for k, v in per_month.items()})
    print(name, 'M5-M7 AUROC %.2f (seeds %s)' % (100 * np.mean(aucs), [round(100 * a, 1) for a in aucs]),
          {k: round(100 * np.mean(v), 1) for k, v in per_month.items()})
# frozen-probe curve across all months (original lp head) for experiment B
from collections import defaultdict
curve = {}
for mth in sorted(set(gn)):
    mk = gn == mth; y = yn[mk].numpy(); s = np.array([m['lp'] for m, k in zip(Mn, mk) if k]); z = np.array([m['zs'] for m, k in zip(Mn, mk) if k])
    if 0 < y.sum() < len(y): curve[mth] = dict(n=int(len(y)), probe=float(roc_auc_score(y, s)), genvideo=float(roc_auc_score(y, z)))
ry = yr.numpy(); curve['M0ref'] = dict(n=len(ry), probe=float(roc_auc_score(ry, [m['lp'] for m in Mr])), genvideo=float(roc_auc_score(ry, [m['zs'] for m in Mr])))
res['frozen_curve'] = curve
for k, v in curve.items(): print(k, v)
json.dump(res, open('refresh_and_monthly.json', 'w'), indent=1)
