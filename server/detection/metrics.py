"""Evaluation metrics: AUROC, bACC, F1, per-group heatmaps.

All functions accept arrays of (scores, labels) where:
  scores: float in [0, 1] — model's AI-probability output
  labels: 0 (real) or 1 (fake)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via trapezoidal integration of the ROC curve.

    Implemented manually to avoid sklearn dep until needed. For large N use
    sklearn.metrics.roc_auc_score instead.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")  # AUROC undefined with single class

    # Sort by score descending
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]

    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")

    # Cumulative TPR/FPR at each unique threshold
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate(([0], tp / P))
    fpr = np.concatenate(([0], fp / N))

    # Trapezoidal area (numpy 2.x renamed np.trapz → np.trapezoid)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels, dtype=int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return float((sensitivity + specificity) / 2)


def f1_score(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels, dtype=int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def per_group_heatmap(
    scores: np.ndarray,
    labels: np.ndarray,
    group_keys: list[str],
    min_count: int = 5,
) -> dict[str, float]:
    """Compute per-group AUROC.

    Used for per-generator and per-platform heatmaps. Groups with fewer
    than `min_count` fake samples return NaN.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    by_group: dict[str, dict] = defaultdict(lambda: {"s": [], "y": []})
    for s, y, g in zip(scores, labels, group_keys):
        by_group[g]["s"].append(s)
        by_group[g]["y"].append(y)

    result: dict[str, float] = {}
    for g, d in by_group.items():
        s_arr = np.array(d["s"])
        y_arr = np.array(d["y"])
        if (y_arr == 1).sum() < min_count or (y_arr == 0).sum() < 1:
            result[g] = float("nan")
        else:
            result[g] = auroc(s_arr, y_arr)
    return result


def all_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "auroc": auroc(scores, labels),
        "bacc": balanced_accuracy(scores, labels),
        "f1": f1_score(scores, labels),
        "n_fake": int((np.asarray(labels) == 1).sum()),
        "n_real": int((np.asarray(labels) == 0).sum()),
    }
