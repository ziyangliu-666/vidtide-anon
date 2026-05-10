"""E3 (NeurIPS 2026 rebuttal Q4): per-generator hardness on the *full* fake pool
using classifier-predicted families for the Unknown bucket.

Inputs:
  - data/ft_features/nsgvd/{train,test}_index.jsonl  (video → label, generator)
  - data/ft_features/{swin,tsm,i3d,slowfast}_{train,test}_X.npy + ids.txt
  - results/bench_5k_{detector}_scores.jsonl         (detector → fake-prob)

Outputs:
  - results/per_generator_full_pool.json
      {
        "labeled":  per-detector × per-family AUROC on the 906-video labeled subset,
        "predicted_hi": per-detector × per-family AUROC on labeled ∪ pred(softmax≥0.8) bucket,
        "ranking_consistency": Spearman ρ between (labeled, predicted_hi) per detector,
      }

Compares the in-paper §6.2 hardness ranking (Dreamina>Sora>Kling) computed
on the 906-video labeled subset to the same ranking computed on the larger
labeled+predicted-high-confidence pool. If rankings hold, the §6.2 conclusion
is robust to the labeled-bias concern.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "ft_features"
RESULTS = REPO / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("e3")

FAMILIES = [
    "kling", "sora", "veo", "dreamina", "runway", "pika",
    "hailuo", "luma", "vidu", "wan", "pixverse", "hunyuan",
    "stable", "animatediff", "ltxv", "modelscope",
]


def family(g):
    if g is None:
        return None
    g_low = g.lower()
    for f in FAMILIES:
        if f in g_low:
            return f
    return g_low


def load_index() -> dict:
    out = {}
    for split in ("train", "test"):
        with (FEAT / "nsgvd" / f"{split}_index.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                out[r["video"]] = r
    return out


def load_backbone(name: str):
    X_tr = np.load(FEAT / f"{name}_train_X.npy")
    X_te = np.load(FEAT / f"{name}_test_X.npy")
    ids_tr = (FEAT / f"{name}_train_ids.txt").read_text().splitlines()
    ids_te = (FEAT / f"{name}_test_ids.txt").read_text().splitlines()
    X = np.concatenate([X_tr, X_te], axis=0).astype(np.float32)
    return X, ids_tr + ids_te


def load_detector_scores() -> dict:
    """Returns {detector_short: {video_id: fake_prob}}."""
    scores = {}
    score_files = sorted(RESULTS.glob("bench_5k_*_scores.jsonl"))
    for f in score_files:
        det = f.name.replace("bench_5k_", "").replace("_scores.jsonl", "")
        d = {}
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                vid = r["video"].rsplit(".", 1)[0]  # strip extension
                d[vid] = float(r["score"])
        scores[det] = d
        log.info("loaded %s: %d scores", det, len(d))
    return scores


def train_classifier_with_per_video(idx, top_n_families: int = 5):
    """Train all4-feature LR classifier (mirroring predict_unknown_generators.py),
    AND save per-video softmax predictions for the Unknown bucket."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    log.info("loading 4 backbone feature sets ...")
    Xs = []
    ref_ids = None
    for bb in ("swin", "tsm", "i3d", "slowfast"):
        X, ids = load_backbone(bb)
        if ref_ids is None:
            ref_ids = ids
        else:
            assert ids == ref_ids, "id mismatch"
        Xs.append(X)
    X_all = np.concatenate(Xs, axis=1)
    log.info("X_all: %s", X_all.shape)

    fake_mask = np.array([idx.get(v, {}).get("label") == 1 for v in ref_ids])
    X_fake = X_all[fake_mask]
    ids_fake = [v for v in ref_ids if idx.get(v, {}).get("label") == 1]
    fams = [family(idx[v].get("generator")) for v in ids_fake]
    log.info("fakes: %d (labeled=%d, unknown=%d)",
             len(X_fake), sum(1 for f in fams if f), sum(1 for f in fams if not f))

    labeled_fams = [f for f in fams if f is not None]
    top5 = [f for f, _ in Counter(labeled_fams).most_common(top_n_families)]
    log.info("top-5 families: %s", top5)

    train_mask = np.array([(f is not None) and (f in top5) for f in fams])
    unknown_mask = np.array([f is None for f in fams])

    X_train = X_fake[train_mask]
    y_train = np.array([fams[i] for i in range(len(fams)) if train_mask[i]])
    X_unknown = X_fake[unknown_mask]
    ids_unknown = [ids_fake[i] for i in range(len(fams)) if unknown_mask[i]]
    ids_labeled = [ids_fake[i] for i in range(len(fams)) if train_mask[i]]
    fams_labeled = [fams[i] for i in range(len(fams)) if train_mask[i]]

    scaler = StandardScaler().fit(X_train)
    X_train_n = scaler.transform(X_train)
    X_unknown_n = scaler.transform(X_unknown)

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_n, y_train)
    proba = clf.predict_proba(X_unknown_n)
    pred = clf.predict(X_unknown_n)
    conf = proba.max(axis=1)
    log.info("predicted: n=%d high-conf(≥0.8)=%d high-conf(≥0.5)=%d",
             len(pred), int((conf >= 0.8).sum()), int((conf >= 0.5).sum()))

    return {
        "classes": list(clf.classes_),
        "ids_labeled": ids_labeled,
        "fams_labeled": fams_labeled,
        "ids_unknown": ids_unknown,
        "pred_unknown": list(pred),
        "conf_unknown": list(conf),
    }


def auroc_per_family(family_to_videos: dict, real_ids: list, scores_per_det: dict) -> dict:
    """For each (detector, family) pair, AUROC = family fakes vs real_ids using
    detector's fake-prob scores. Returns nested dict[detector][family] -> {n_fake, auroc}."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for det, sd in scores_per_det.items():
        out[det] = {}
        # Reals (positives=0)
        real_p = [sd[v] for v in real_ids if v in sd]
        n_real = len(real_p)
        if n_real == 0:
            log.warning("[%s] no real scores", det)
            continue
        for fam, fake_ids in family_to_videos.items():
            fake_p = [sd[v] for v in fake_ids if v in sd]
            n_fake = len(fake_p)
            if n_fake < 5:
                out[det][fam] = {"n_fake": n_fake, "n_real": n_real, "auroc": None}
                continue
            y = np.array([0] * n_real + [1] * n_fake)
            p = np.array(real_p + fake_p)
            try:
                a = roc_auc_score(y, p)
            except Exception:
                a = None
            out[det][fam] = {"n_fake": n_fake, "n_real": n_real, "auroc": float(a) if a is not None else None}
    return out


def main():
    idx = load_index()
    log.info("loaded index: %d videos", len(idx))

    real_ids = [v for v, r in idx.items() if r.get("label") == 0]
    log.info("real videos in eval index: %d", len(real_ids))

    # ---------- Labeled-only family→video map ----------
    fam_labeled_videos = defaultdict(list)
    for v, r in idx.items():
        if r.get("label") != 1:
            continue
        f = family(r.get("generator"))
        if f is not None:
            fam_labeled_videos[f].append(v)
    fam_labeled_videos = {k: v for k, v in fam_labeled_videos.items()
                          if k in {"kling", "sora", "veo", "dreamina", "runway"}}
    for f, vs in fam_labeled_videos.items():
        log.info("labeled %-10s %d", f, len(vs))

    # ---------- Train classifier + collect Unknown predictions ----------
    pred = train_classifier_with_per_video(idx)

    fam_predicted_videos = {f: list(fam_labeled_videos[f]) for f in fam_labeled_videos}
    n_added = {f: 0 for f in fam_labeled_videos}
    for vid, fam, c in zip(pred["ids_unknown"], pred["pred_unknown"], pred["conf_unknown"]):
        if c >= 0.8 and fam in fam_predicted_videos:
            fam_predicted_videos[fam].append(vid)
            n_added[fam] += 1
    for f in fam_predicted_videos:
        log.info("predicted-hi %-10s labeled=%d +pred=%d → total=%d",
                 f, len(fam_labeled_videos[f]), n_added[f], len(fam_predicted_videos[f]))

    # ---------- Detector scores ----------
    scores = load_detector_scores()
    if not scores:
        log.error("no detector score files found")
        sys.exit(1)

    # ---------- AUROC: labeled-subset vs predicted-high-confidence pool ----------
    log.info("\n=== labeled-only family AUROC ===")
    auc_labeled = auroc_per_family(fam_labeled_videos, real_ids, scores)
    log.info("\n=== labeled+predicted-hi family AUROC ===")
    auc_pred = auroc_per_family(fam_predicted_videos, real_ids, scores)

    # ---------- Ranking consistency (per detector) ----------
    from scipy.stats import spearmanr  # type: ignore
    ranking = {}
    families_order = ["kling", "sora", "veo", "dreamina", "runway"]
    for det in scores:
        a_lab = [auc_labeled.get(det, {}).get(f, {}).get("auroc") for f in families_order]
        a_pre = [auc_pred.get(det, {}).get(f, {}).get("auroc") for f in families_order]
        # mask out None
        pairs = [(x, y) for x, y in zip(a_lab, a_pre) if x is not None and y is not None]
        if len(pairs) < 3:
            ranking[det] = {"rho": None, "n": len(pairs)}
            continue
        xs, ys = zip(*pairs)
        rho, p = spearmanr(xs, ys)
        ranking[det] = {"rho": float(rho), "p": float(p), "n": len(pairs)}

    out = {
        "labeled_subset_size": {f: len(fam_labeled_videos[f]) for f in fam_labeled_videos},
        "predicted_hi_added": n_added,
        "predicted_hi_total": {f: len(fam_predicted_videos[f]) for f in fam_predicted_videos},
        "auroc_labeled": auc_labeled,
        "auroc_predicted_hi": auc_pred,
        "spearman_per_detector": ranking,
    }

    out_path = RESULTS / "per_generator_full_pool.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", out_path)

    # Summary table
    log.info("\n=== summary: per-detector ranking on labeled vs predicted-hi pool ===")
    log.info("%-30s | %s", "detector", " | ".join([f"{f:>10s} (lab/pred)" for f in families_order]))
    for det in sorted(scores):
        cells = []
        for f in families_order:
            a = auc_labeled.get(det, {}).get(f, {}).get("auroc")
            b = auc_pred.get(det, {}).get(f, {}).get("auroc")
            a_s = f"{a:.3f}" if a is not None else "  -  "
            b_s = f"{b:.3f}" if b is not None else "  -  "
            cells.append(f"{a_s}/{b_s}")
        rho = ranking[det].get("rho")
        rho_s = f"ρ={rho:+.2f}" if rho is not None else "ρ=N/A"
        log.info("%-30s | %s | %s", det, " | ".join(cells), rho_s)


if __name__ == "__main__":
    main()
