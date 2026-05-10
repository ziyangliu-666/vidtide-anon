"""Supervised generator-family classifier (X1 pivot, NeurIPS 2026 rebuttal Q1).

Train a linear classifier on the labeled-fake subset, apply to the unknown-fake bucket,
and report the predicted generator-family distribution.

Output: results/predict_unknown_generators.json
        figs/cluster_cache/predict_distribution.png
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "ft_features"
RESULTS = REPO / "results"
FIGS = REPO / "data" / "cluster_cache" / "figs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")


# ---------- family normalization ----------

FAMILIES = [
    "kling", "sora", "veo", "dreamina", "runway", "pika",
    "hailuo", "luma", "vidu", "wan", "pixverse", "hunyuan",
    "stable", "animatediff", "ltxv", "modelscope",
]


def family(g: str | None) -> str | None:
    if g is None:
        return None
    g_low = g.lower()
    for f in FAMILIES:
        if f in g_low:
            return f
    return g_low


# ---------- feature loading ----------


def load_backbone(name: str) -> tuple[np.ndarray, list[str]]:
    X_tr = np.load(FEAT / f"{name}_train_X.npy")
    X_te = np.load(FEAT / f"{name}_test_X.npy")
    ids_tr = (FEAT / f"{name}_train_ids.txt").read_text().splitlines()
    ids_te = (FEAT / f"{name}_test_ids.txt").read_text().splitlines()
    X = np.concatenate([X_tr, X_te], axis=0).astype(np.float32)
    return X, ids_tr + ids_te


def load_index() -> dict[str, dict]:
    """video_id -> {label, generator, platform}"""
    out = {}
    for split in ("train", "test"):
        with open(FEAT / "nsgvd" / f"{split}_index.jsonl") as f:
            for line in f:
                r = json.loads(line)
                out[r["video"]] = r
    return out


# ---------- main ----------


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    idx = load_index()
    log.info("NSG-VD index: %d entries", len(idx))

    # Feature combinations to try
    feature_sets = {
        "swin": ["swin"],
        "tsm": ["tsm"],
        "i3d": ["i3d"],
        "slowfast": ["slowfast"],
        "swin+tsm": ["swin", "tsm"],
        "all4": ["swin", "tsm", "i3d", "slowfast"],
    }

    all_results = {}

    for fset_name, backbones in feature_sets.items():
        log.info("\n========== feature set: %s ==========", fset_name)

        # Build feature matrix; concat across backbones (must align ids)
        Xs = []
        ids = None
        for bb in backbones:
            X, bb_ids = load_backbone(bb)
            if ids is None:
                ids = bb_ids
            else:
                assert ids == bb_ids, f"id mismatch between backbones"
            Xs.append(X)
        X_all = np.concatenate(Xs, axis=1)
        log.info("  feature dim: %d", X_all.shape[1])

        # Filter to fakes only
        fake_mask = np.array([idx.get(v, {}).get("label") == 1 for v in ids])
        X_fake = X_all[fake_mask]
        gens_fake = [idx[v].get("generator") for v in ids if idx.get(v, {}).get("label") == 1]
        fams_fake = [family(g) for g in gens_fake]
        ids_fake = [v for v in ids if idx.get(v, {}).get("label") == 1]
        log.info("  fakes: %d (labeled=%d, unknown=%d)",
                 len(X_fake), sum(1 for f in fams_fake if f), sum(1 for f in fams_fake if not f))

        # Top-5 family restriction for training
        labeled_idx = [i for i, f in enumerate(fams_fake) if f is not None]
        labeled_fams = [fams_fake[i] for i in labeled_idx]
        top5 = [f for f, _ in Counter(labeled_fams).most_common(5)]
        keep_mask = np.array([fams_fake[i] in top5 for i in range(len(fams_fake))])
        train_mask = keep_mask & np.array([f is not None for f in fams_fake])
        unknown_mask = np.array([f is None for f in fams_fake])

        X_train = X_fake[train_mask]
        y_train = np.array([fams_fake[i] for i in range(len(fams_fake)) if train_mask[i]])
        X_unknown = X_fake[unknown_mask]
        ids_unknown = [ids_fake[i] for i in range(len(fams_fake)) if unknown_mask[i]]
        log.info("  train (top-5 families: %s): %d samples", top5, len(X_train))
        log.info("  unknown to predict: %d samples", len(X_unknown))

        # Standardize using ONLY training data
        scaler = StandardScaler().fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_unknown_n = scaler.transform(X_unknown)

        # 5-fold CV with class-balanced LR
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        cv_accs = []
        cv_top3 = []
        for tr, va in skf.split(X_train_n, y_train):
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                     solver="lbfgs", n_jobs=-1)
            clf.fit(X_train_n[tr], y_train[tr])
            pred = clf.predict(X_train_n[va])
            cv_accs.append((pred == y_train[va]).mean())
            proba = clf.predict_proba(X_train_n[va])
            top3 = np.argsort(-proba, axis=1)[:, :3]
            class_idx = {c: i for i, c in enumerate(clf.classes_)}
            true_idx = np.array([class_idx[y] for y in y_train[va]])
            cv_top3.append(np.mean([t in top3[i] for i, t in enumerate(true_idx)]))
        log.info("  CV top-1 acc: %.3f ± %.3f", np.mean(cv_accs), np.std(cv_accs))
        log.info("  CV top-3 acc: %.3f ± %.3f", np.mean(cv_top3), np.std(cv_top3))

        # Detailed per-class report on a single fold for diagnostics
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", n_jobs=-1)
        cv_iter = list(skf.split(X_train_n, y_train))
        tr0, va0 = cv_iter[0]
        clf.fit(X_train_n[tr0], y_train[tr0])
        pred0 = clf.predict(X_train_n[va0])
        log.info("  per-class (fold-0):\n%s",
                 classification_report(y_train[va0], pred0, zero_division=0))

        # Train final classifier on all labeled top-5
        clf_final = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                       solver="lbfgs", n_jobs=-1)
        clf_final.fit(X_train_n, y_train)

        # Predict on unknown
        y_pred = clf_final.predict(X_unknown_n)
        y_proba = clf_final.predict_proba(X_unknown_n)
        conf = y_proba.max(axis=1)

        # All predictions
        log.info("  predicted distribution on %d unknown (all):", len(y_pred))
        all_dist = Counter(y_pred)
        for f in top5:
            c = all_dist.get(f, 0)
            log.info("    %-10s %5d  (%.1f%%)", f, c, 100 * c / len(y_pred))

        # High-confidence predictions
        hi = conf >= 0.5
        log.info("  high-confidence (max prob >= 0.5): %d/%d (%.1f%%)",
                 hi.sum(), len(y_pred), 100 * hi.mean())
        if hi.sum() > 0:
            hi_dist = Counter(y_pred[hi])
            for f in top5:
                c = hi_dist.get(f, 0)
                log.info("    %-10s %5d  (%.1f%%)", f, c, 100 * c / hi.sum())

        all_results[fset_name] = {
            "feature_dim": int(X_all.shape[1]),
            "n_train": int(len(X_train)),
            "n_unknown": int(len(X_unknown)),
            "top5_families": top5,
            "cv_top1_acc_mean": float(np.mean(cv_accs)),
            "cv_top1_acc_std": float(np.std(cv_accs)),
            "cv_top3_acc_mean": float(np.mean(cv_top3)),
            "cv_top3_acc_std": float(np.std(cv_top3)),
            "predicted_distribution_all": {k: int(v) for k, v in all_dist.items()},
            "predicted_distribution_high_conf": {k: int(v) for k, v in (Counter(y_pred[hi]).items() if hi.sum() else [])},
            "n_high_conf": int(hi.sum()),
            "mean_confidence": float(conf.mean()),
            "n_classes_present_high_conf": int(len({f for f in y_pred[hi]})) if hi.sum() else 0,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "predict_unknown_generators.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    log.info("\nresults saved: %s", out_path)

    # Summary table
    log.info("\n========== SUMMARY ==========")
    log.info("%-12s | %-9s | %-9s | %-12s | %s", "feature", "CV-top1", "CV-top3", "high-conf %", "families ≥5%")
    for fset, r in all_results.items():
        hi_dist = r["predicted_distribution_high_conf"]
        total_hi = sum(hi_dist.values()) or 1
        big_fams = [f for f, c in hi_dist.items() if c / total_hi >= 0.05]
        log.info("%-12s | %.3f±%.3f | %.3f±%.3f | %.1f%%       | %d (%s)",
                 fset, r["cv_top1_acc_mean"], r["cv_top1_acc_std"],
                 r["cv_top3_acc_mean"], r["cv_top3_acc_std"],
                 100 * r["n_high_conf"] / r["n_unknown"],
                 len(big_fams), ", ".join(big_fams))


if __name__ == "__main__":
    main()
