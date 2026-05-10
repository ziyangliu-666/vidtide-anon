"""Try clustering with detector backbone features (swin/tsm/i3d/slowfast/nsgvd)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
META = REPO / "data" / "cluster_cache" / "meta.json"
FEAT = REPO / "data" / "ft_features"

with open(META) as f:
    m = json.load(f)
gen_lookup = dict(zip(m["ids"], m["gens"]))


def cluster_one(name, X, gens):
    n_known = len(set(g for g in gens if g))
    n_lab = sum(1 for g in gens if g)
    Xn = StandardScaler().fit_transform(X.astype(np.float32))
    print(f"  N={len(X)}, dim={X.shape[1]}, labeled={n_lab}, n_known_gens={n_known}")

    lab_mask = np.array([g is not None for g in gens])
    gen_arr = np.array([g if g else "" for g in gens])

    sample_n = min(3000, len(Xn))
    sample_idx = np.random.RandomState(0).choice(len(Xn), size=sample_n, replace=False)

    for k in [3, 5, 7, 10, 15, 20, 24]:
        km = KMeans(n_clusters=k, random_state=0, n_init=5).fit(Xn)
        sil = silhouette_score(Xn[sample_idx], km.labels_[sample_idx])
        sub_l = km.labels_[lab_mask]
        sub_g = gen_arr[lab_mask]
        purity = sum(
            np.unique(sub_g[sub_l == c], return_counts=True)[1].max()
            for c in np.unique(sub_l) if (sub_l == c).any()
        ) / len(sub_l)
        nmi = normalized_mutual_info_score(sub_g, sub_l)
        print(f"  k={k:2d}  sil={sil:.3f}  purity={purity:.3f}  nmi={nmi:.3f}")

    if lab_mask.sum() > 50 and n_known >= 2:
        Xl = Xn[lab_mask]
        gl = np.array([g for g in gens if g])
        km2 = KMeans(n_clusters=min(n_known, len(Xl) - 1), random_state=0, n_init=5).fit(Xl)
        sanity = sum(
            np.unique(gl[km2.labels_ == c], return_counts=True)[1].max()
            for c in np.unique(km2.labels_) if (km2.labels_ == c).any()
        ) / len(Xl)
        print(f"  sanity_k={n_known} purity={sanity:.3f}  (target>0.6)")


for backbone in ["swin", "tsm", "i3d", "slowfast"]:
    print(f"\n========== {backbone} ==========")
    X_tr = np.load(FEAT / f"{backbone}_train_X.npy")
    X_te = np.load(FEAT / f"{backbone}_test_X.npy")
    with open(FEAT / f"{backbone}_train_ids.txt") as f:
        ids_tr = [l.strip() for l in f]
    with open(FEAT / f"{backbone}_test_ids.txt") as f:
        ids_te = [l.strip() for l in f]
    X = np.concatenate([X_tr, X_te], axis=0)
    ids = ids_tr + ids_te
    keep = [i for i, v in enumerate(ids) if v in gen_lookup]
    X = X[keep]
    gens = [gen_lookup[ids[i]] for i in keep]
    cluster_one(backbone, X, gens)
