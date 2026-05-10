"""Deeper post-hoc analysis of the supervised generator-family classifier.

Generates the numbers + figure that go into the rebuttal appendix.
Outputs:
  results/predict_unknown_analysis.json
  results/figs/predict_unknown.pdf  (figure for paper)
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
FIGS = RESULTS / "figs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze")

FAMILIES = [
    "kling", "sora", "veo", "dreamina", "runway", "pika",
    "hailuo", "luma", "vidu", "wan", "pixverse", "hunyuan",
    "stable", "animatediff", "ltxv", "modelscope",
]


def family(g):
    if g is None:
        return None
    g = g.lower()
    for f in FAMILIES:
        if f in g:
            return f
    return g


def load_backbone(name):
    X_tr = np.load(FEAT / f"{name}_train_X.npy")
    X_te = np.load(FEAT / f"{name}_test_X.npy")
    ids_tr = (FEAT / f"{name}_train_ids.txt").read_text().splitlines()
    ids_te = (FEAT / f"{name}_test_ids.txt").read_text().splitlines()
    X = np.concatenate([X_tr, X_te], axis=0).astype(np.float32)
    return X, ids_tr + ids_te


def load_index():
    out = {}
    for split in ("train", "test"):
        with open(FEAT / "nsgvd" / f"{split}_index.jsonl") as f:
            for line in f:
                r = json.loads(line)
                out[r["video"]] = r
    return out


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    idx = load_index()

    # Best feature set: all4
    backbones = ["swin", "tsm", "i3d", "slowfast"]
    Xs = []
    ids = None
    for bb in backbones:
        X, bb_ids = load_backbone(bb)
        if ids is None:
            ids = bb_ids
        Xs.append(X)
    X_all = np.concatenate(Xs, axis=1)
    log.info("loaded all4 features: %s", X_all.shape)

    # Filter to fakes
    fake_mask = np.array([idx.get(v, {}).get("label") == 1 for v in ids])
    X_fake = X_all[fake_mask]
    ids_fake = [v for v in ids if idx.get(v, {}).get("label") == 1]
    gens_raw = [idx[v].get("generator") for v in ids_fake]
    fams = [family(g) for g in gens_raw]
    plats = [idx[v].get("platform", "unknown") for v in ids_fake]

    # Top-5 family training set
    labeled_fams = [f for f in fams if f]
    top5 = [f for f, _ in Counter(labeled_fams).most_common(5)]
    log.info("top-5 families: %s", top5)

    train_mask = np.array([(f in top5) for f in fams])
    unknown_mask = np.array([f is None for f in fams])
    X_train = X_fake[train_mask]
    y_train = np.array([fams[i] for i in range(len(fams)) if train_mask[i]])
    plat_train = [plats[i] for i in range(len(fams)) if train_mask[i]]
    X_unknown = X_fake[unknown_mask]
    plat_unknown = [plats[i] for i in range(len(fams)) if unknown_mask[i]]

    log.info("N_train=%d, N_unknown=%d", len(X_train), len(X_unknown))

    # Standardize using only train
    scaler = StandardScaler().fit(X_train)
    X_train_n = scaler.transform(X_train)
    X_unknown_n = scaler.transform(X_unknown)

    # ---------- 5-fold CV with detailed per-class metrics ----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_pred = np.empty_like(y_train)
    for tr, va in skf.split(X_train_n, y_train):
        clf = LogisticRegression(C=1.0, max_iter=2000,
                                 class_weight="balanced", solver="lbfgs")
        clf.fit(X_train_n[tr], y_train[tr])
        all_pred[va] = clf.predict(X_train_n[va])
    cv_acc_top1 = (all_pred == y_train).mean()
    log.info("5-fold OOF top-1 accuracy: %.4f", cv_acc_top1)

    # per-family P/R/F1 (5-fold OOF)
    cls_report = classification_report(y_train, all_pred,
                                       labels=top5, output_dict=True, zero_division=0)
    log.info("per-family OOF P/R/F1:\n%s",
             classification_report(y_train, all_pred, labels=top5, zero_division=0))

    cm = confusion_matrix(y_train, all_pred, labels=top5)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    log.info("confusion matrix (rows=true, cols=pred):\n%s", cm)

    # ---------- Final model + Unknown predictions ----------
    clf_final = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", solver="lbfgs")
    clf_final.fit(X_train_n, y_train)
    y_pred = clf_final.predict(X_unknown_n)
    y_proba = clf_final.predict_proba(X_unknown_n)
    conf = y_proba.max(axis=1)

    # by-platform predicted distribution
    by_plat = {}
    for p in sorted(set(plat_unknown)):
        mask = np.array([pp == p for pp in plat_unknown])
        if mask.sum() < 30:
            continue
        dist = Counter(y_pred[mask])
        by_plat[p] = {"n": int(mask.sum()),
                      **{f: int(dist.get(f, 0)) for f in top5}}
    log.info("by-platform predicted distribution: %s", by_plat)

    # Confidence stratification
    conf_bins = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    conf_dist = {}
    for lo, hi in conf_bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        d = Counter(y_pred[m])
        conf_dist[f"{lo:.1f}-{hi:.2f}"] = {
            "n": int(m.sum()),
            "frac": float(m.mean()),
            **{f: int(d.get(f, 0)) for f in top5},
        }
    log.info("confidence stratification: %s", conf_dist)

    # Compare labeled vs predicted (overall)
    lab_dist = {f: int(sum(1 for y in y_train if y == f)) for f in top5}
    pred_dist_all = {f: int(sum(1 for y in y_pred if y == f)) for f in top5}
    pred_dist_high = {f: int(sum(1 for y, c in zip(y_pred, conf) if y == f and c >= 0.5))
                      for f in top5}

    # Save analysis JSON
    analysis = {
        "feature_set": "all4 (swin+tsm+i3d+slowfast)",
        "feature_dim": int(X_all.shape[1]),
        "top5_families": top5,
        "n_train_labeled": int(len(X_train)),
        "n_unknown_predicted": int(len(X_unknown)),
        "cv_top1_acc_oof": float(cv_acc_top1),
        "per_family_metrics": {
            f: {
                "precision": float(cls_report[f]["precision"]),
                "recall": float(cls_report[f]["recall"]),
                "f1_score": float(cls_report[f]["f1-score"]),
                "support": int(cls_report[f]["support"]),
            }
            for f in top5
        },
        "macro_avg": {
            "precision": float(cls_report["macro avg"]["precision"]),
            "recall": float(cls_report["macro avg"]["recall"]),
            "f1_score": float(cls_report["macro avg"]["f1-score"]),
        },
        "confusion_matrix": {
            "labels": top5,
            "raw": cm.tolist(),
            "row_normalized": cm_normalized.tolist(),
        },
        "labeled_distribution": lab_dist,
        "predicted_distribution_all": pred_dist_all,
        "predicted_distribution_high_conf": pred_dist_high,
        "n_high_conf": int((conf >= 0.5).sum()),
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
        "confidence_stratification": conf_dist,
        "by_platform": by_plat,
    }
    out_path = RESULTS / "predict_unknown_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    log.info("saved: %s", out_path)

    # ---------- Figure: distribution comparison + confusion + confidence histogram ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
    })

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})

    # Panel A: distribution comparison
    ax = axes[0]
    x = np.arange(len(top5))
    width = 0.32
    lab_total = sum(lab_dist.values())
    pred_total = sum(pred_dist_all.values())
    lab_pct = [100 * lab_dist[f] / lab_total for f in top5]
    pred_pct = [100 * pred_dist_all[f] / pred_total for f in top5]
    ax.bar(x - width / 2, lab_pct, width, color="#2563EB",
           label=f"Labeled (N={lab_total:,})", edgecolor="none")
    ax.bar(x + width / 2, pred_pct, width, color="#10B981",
           label=f"Predicted on Unknown (N={pred_total:,})", edgecolor="none")
    for xi, v in zip(x - width / 2, lab_pct):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=6, color="#2563EB")
    for xi, v in zip(x + width / 2, pred_pct):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=6, color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(top5, fontsize=7.5)
    ax.set_ylabel("share (%)", fontsize=7.5)
    ax.set_title("(a) Family share: labeled vs predicted Unknown",
                 fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_ylim(0, max(max(lab_pct), max(pred_pct)) * 1.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: confusion matrix (row-normalized)
    ax = axes[1]
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top5)))
    ax.set_yticks(range(len(top5)))
    ax.set_xticklabels(top5, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(top5, fontsize=7)
    ax.set_xlabel("predicted", fontsize=7.5)
    ax.set_ylabel("true", fontsize=7.5)
    ax.set_title("(b) 5-fold CV confusion (row-normalized)",
                 fontsize=8.5, fontweight="bold")
    for i in range(len(top5)):
        for j in range(len(top5)):
            v = cm_normalized[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v > 0.5 else "#1F2937")

    # Panel C: confidence histogram
    ax = axes[2]
    ax.hist(conf, bins=30, color="#7C3AED", edgecolor="none", alpha=0.85)
    ax.axvline(0.5, color="#DC2626", linestyle="--", linewidth=0.8)
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "high-conf\nthreshold",
            fontsize=6, color="#DC2626")
    ax.set_xlabel("max softmax", fontsize=7.5)
    ax.set_ylabel("Unknown videos", fontsize=7.5)
    ax.set_title(f"(c) Prediction confidence (mean={conf.mean():.2f})",
                 fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.6)
    out_pdf = FIGS / "predict_unknown.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    out_png = FIGS / "predict_unknown.png"
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=160)
    log.info("figure saved: %s and .png", out_pdf)


if __name__ == "__main__":
    main()
