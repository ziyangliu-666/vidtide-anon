"""Dump every number the NeurIPS paper tables need, in copy-paste-ready form.

Outputs go to stdout and ``results/paper_numbers.json`` for durability.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DB = REPO / "data" / "vidtide.db"


def _fmt(x, prec=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * float(x):.{prec}f}"


def load(name):
    return json.load(open(RESULTS / name))


def load_jsonl(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]


# ==========================================================
# TABLE 4: tab:gap - per-detector AUROC/bACC/F1 on bench M0
# ==========================================================
def table4_gap():
    files = {
        "npr_pika":      "bench_5k_pika_crafter.json",
        "npr_crafter":   "bench_5k_pika_crafter.json",
        "tall_pika":     "bench_5k_pika_crafter.json",
        "tall_crafter":  "bench_5k_pika_crafter.json",
        "stil_pika":     "bench_5k_stil_pika.json",
        "stil_crafter":  "bench_5k_stil_crafter.json",
        "demamba_pika":  "bench_5k_demamba_pika.json",
        "demamba_crafter": "bench_5k_demamba_crafter.json",
        "nsgvd_pika":    "bench_5k_nsgvd_pika.json",
        "clip_zero_shot": "bench_5k_clip_zero_shot.json",
    }
    out = {}
    for det, f in files.items():
        d = load(f)[det]
        out[det] = {
            "auroc": d["auroc"], "bacc": d["bacc"], "f1": d["f1"],
            "n_fake": d["n_fake"], "n_real": d["n_real"],
        }
    return out


# ==========================================================
# TABLE 14 expansion: per-generator × per-detector AUROC+TPR@5
# Need TPR@5 from JSONL scores: threshold derived from each
# detector's FULL real set, then applied to generator-slice fakes.
# ==========================================================
def load_metadata():
    con = sqlite3.connect(DB)
    meta = {
        vid: (plat, gen or "_unlabeled", label)
        for vid, plat, gen, label in con.execute(
            "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator,''),''), label FROM videos"
        )
    }
    con.close()
    return meta


DETECTOR_TO_JSONL = {
    "demamba_pika": "bench_5k_demamba_pika_scores.jsonl",
    "demamba_crafter": "bench_5k_demamba_crafter_scores.jsonl",
    "npr_pika": "bench_5k_pika_crafter_scores.jsonl",
    "npr_crafter": "bench_5k_pika_crafter_scores.jsonl",
    "tall_pika": "bench_5k_pika_crafter_scores.jsonl",
    "tall_crafter": "bench_5k_pika_crafter_scores.jsonl",
    "stil_pika": "bench_5k_stil_pika_scores.jsonl",
    "stil_crafter": "bench_5k_stil_crafter_scores.jsonl",
    "nsgvd_pika": "bench_5k_nsgvd_pika_scores.jsonl",
    "clip_zero_shot": "bench_5k_clip_zero_shot_scores.jsonl",
}


def vid_from_path(p):
    # scores store "video": "/path/to/uuid.mp4"
    return Path(p).stem


def per_slice_auroc_tpr(detector, meta, target_fpr=0.05):
    """Return dict: slice_key -> (auroc, tpr@5, f1@0.5, n_fake).

    F1@0.5 is computed by pairing each slice's fakes with the detector's full
    real pool (reals don't partition by generator, so this is the only
    coherent choice; per-platform reruns reuse the same convention for
    consistency with TPR@5)."""
    recs = load_jsonl(DETECTOR_TO_JSONL[detector])
    # filter to this detector only (jsonl files can contain many detectors)
    recs = [r for r in recs if r.get("detector") == detector]
    fakes_by_gen = defaultdict(list)
    fakes_by_plat = defaultdict(list)
    reals = []
    for r in recs:
        vid = vid_from_path(r["video"])
        m = meta.get(vid)
        if m is None:
            continue
        plat, gen, _ = m
        if r["label"] == 1:  # fake
            fakes_by_gen[gen].append(r["score"])
            fakes_by_plat[plat].append(r["score"])
        else:
            reals.append(r["score"])
    reals = np.asarray(reals)
    if len(reals) == 0:
        return {}
    thr = float(np.quantile(reals, 1.0 - target_fpr))
    fp05_global = int((reals >= 0.5).sum())
    result = {"_n_real_global": len(reals)}

    def _slice_metrics(arr):
        # AUROC vs global reals
        y = np.r_[np.ones(len(arr)), np.zeros(len(reals))]
        s = np.r_[arr, reals]
        order = np.argsort(-s, kind="mergesort"); ys = y[order]
        tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
        P, N = float((ys == 1).sum()), float((ys == 0).sum())
        if P == 0 or N == 0:
            auroc = float("nan")
        else:
            tpr = np.concatenate(([0.], tp / P)); fpr = np.concatenate(([0.], fp / N))
            trap = getattr(np, "trapezoid", None) or np.trapz
            auroc = float(trap(tpr, fpr))
        tpr5 = float((arr >= thr).mean())
        # F1@0.5: slice fakes vs global reals
        tp05 = int((arr >= 0.5).sum())
        fn05 = int((arr < 0.5).sum())
        if tp05 == 0:
            f1 = 0.0
        else:
            prec = tp05 / (tp05 + fp05_global)
            rec = tp05 / (tp05 + fn05)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return auroc, tpr5, float(f1)

    # per-generator
    for gen, scores in fakes_by_gen.items():
        arr = np.asarray(scores)
        if len(arr) < 10:
            continue
        auroc, tpr5, f1 = _slice_metrics(arr)
        result[f"gen::{gen}"] = (auroc, tpr5, f1, len(arr))
    # per-platform
    for plat, scores in fakes_by_plat.items():
        arr = np.asarray(scores)
        if len(arr) < 30:
            continue
        auroc, tpr5, f1 = _slice_metrics(arr)
        result[f"plat::{plat}"] = (auroc, tpr5, f1, len(arr))
    return result


# ==========================================================
# TABLE: Per-platform × per-detector (9 × 3 × {AUROC,bACC,F1})
# Need bACC and F1 recomputed per slice.
# ==========================================================
def per_platform_triple(detector, meta, target_fpr=0.05):
    recs = load_jsonl(DETECTOR_TO_JSONL[detector])
    recs = [r for r in recs if r.get("detector") == detector]
    fakes_by_plat = defaultdict(list)
    reals_global = []
    reals_by_plat = defaultdict(list)
    for rec in recs:
        vid = vid_from_path(rec["video"])
        m = meta.get(vid)
        if m is None:
            continue
        plat = m[0]
        if rec["label"] == 1:
            fakes_by_plat[plat].append(rec["score"])
        else:
            reals_global.append(rec["score"])
            reals_by_plat[plat].append(rec["score"])
    # Global TPR@5 threshold (same convention as per_slice_auroc_tpr)
    rg = np.asarray(reals_global)
    thr_global = float(np.quantile(rg, 1.0 - target_fpr)) if len(rg) else float("nan")
    result = {}
    for plat, fakes in fakes_by_plat.items():
        nf = len(fakes)
        if nf < 30:
            continue
        # same-platform reals if available, else global pool
        if len(reals_by_plat[plat]) >= 30:
            rpool = reals_by_plat[plat]; real_src = "same-plat"
        else:
            rpool = reals_global; real_src = "global"
        nr = len(rpool)
        f = np.asarray(fakes); r = np.asarray(rpool)
        # AUROC
        y = np.r_[np.ones(nf), np.zeros(nr)]; s = np.r_[f, r]
        order = np.argsort(-s, kind="mergesort"); ys = y[order]
        tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
        tpr = np.concatenate(([0.], tp / nf)); fpr = np.concatenate(([0.], fp / nr))
        trap = getattr(np, "trapezoid", None) or np.trapz
        auroc = float(trap(tpr, fpr))
        # bACC @ 0.5
        tpr05 = float((f >= 0.5).mean()); tnr05 = float((r < 0.5).mean())
        bacc = 0.5 * (tpr05 + tnr05)
        # F1 @ 0.5
        tp05 = int((f >= 0.5).sum()); fp05 = int((r >= 0.5).sum()); fn05 = int((f < 0.5).sum())
        if tp05 == 0: f1 = 0.0
        else:
            prec = tp05 / (tp05 + fp05); rec = tp05 / (tp05 + fn05)
            f1 = 2 * prec * rec / (prec + rec)
        # TPR@5 using global-reals threshold
        tpr5 = float((f >= thr_global).mean()) if not np.isnan(thr_global) else float("nan")
        result[plat] = {"auroc": auroc, "bacc": bacc, "f1": f1, "tpr5": tpr5,
                        "n_fake": nf, "n_real": nr, "real_src": real_src}
    return result


# ==========================================================
# TABLE: M0 composition (duration/resolution/fps/label-source)
# ==========================================================
def m0_composition():
    con = sqlite3.connect(DB)
    c = con.cursor()
    out = {"by_platform": {}, "by_label_source": {}, "overall": {}}
    # Per-platform duration/resolution stats on active eval pool (status='active')
    rows = c.execute("""
        SELECT source_platform, label,
               COUNT(*) as n,
               AVG(duration_sec), MIN(duration_sec), MAX(duration_sec),
               AVG(resolution_w), AVG(resolution_h), AVG(fps)
        FROM videos
        WHERE status = 'filtered' AND duration_sec IS NOT NULL
        GROUP BY source_platform, label
    """).fetchall()
    for plat, lab, n, dur, dmin, dmax, w, h, fps in rows:
        out["by_platform"].setdefault(plat, {})[lab] = {
            "n": n, "dur_mean": dur, "dur_min": dmin, "dur_max": dmax,
            "w": w, "h": h, "fps": fps,
        }
    rows = c.execute("""
        SELECT label_source, COUNT(*) AS n, AVG(label_confidence) AS conf
        FROM videos WHERE status = 'filtered' GROUP BY label_source
    """).fetchall()
    for src, n, conf in rows:
        out["by_label_source"][src or "(null)"] = {"n": n, "confidence": conf}
    # totals
    n_total = c.execute(
        "SELECT COUNT(*) FROM videos WHERE status = 'filtered'"
    ).fetchone()[0]
    out["overall"]["n_total"] = n_total
    con.close()
    return out


def main():
    print("=" * 60); print("TABLE 4: bench-Pika / bench-Crafter triple-metric"); print("=" * 60)
    t4 = table4_gap()
    for det, v in t4.items():
        print(f"  {det:<20} AUROC={_fmt(v['auroc'])}  bACC={_fmt(v['bacc'])}  F1={_fmt(v['f1'],2)}  (n_f={v['n_fake']} n_r={v['n_real']})")

    print()
    print("=" * 60); print("META JOIN: loading vidtide.db ..."); print("=" * 60)
    meta = load_metadata()
    print(f"  {len(meta)} videos loaded")

    print()
    print("=" * 60); print("Per-detector per-generator AUROC/TPR@5 (from JSONL)"); print("=" * 60)
    per_det_slice = {}
    for det in DETECTOR_TO_JSONL:
        try:
            per_det_slice[det] = per_slice_auroc_tpr(det, meta)
            print(f"  {det:<20} slices: {len(per_det_slice[det])-1}")
        except Exception as e:
            print(f"  {det:<20} ERROR {e}")

    print()
    print("=" * 60); print("Per-detector per-platform triple-metric"); print("=" * 60)
    per_det_plat = {}
    for det in DETECTOR_TO_JSONL:
        try:
            per_det_plat[det] = per_platform_triple(det, meta)
            for plat, v in per_det_plat[det].items():
                print(f"  {det:<20} {plat:<10} AUROC={_fmt(v['auroc'])}  bACC={_fmt(v['bacc'])}  F1={_fmt(v['f1'],2)}  TPR@5={_fmt(v['tpr5'])}  (nf={v['n_fake']} nr={v['n_real']})")
        except Exception as e:
            print(f"  {det:<20} ERROR {e}")

    print()
    print("=" * 60); print("M0 DB composition"); print("=" * 60)
    comp = m0_composition()
    print(f"  Active-pool total: {comp['overall']['n_total']}")
    for plat, labs in comp["by_platform"].items():
        for lab, v in labs.items():
            dm = v['dur_mean'] or 0; dmin = v['dur_min'] or 0; dmax = v['dur_max'] or 0
            w = int(v['w'] or 0); h = int(v['h'] or 0); fps = v['fps'] or 0
            print(f"  {plat:<10} {lab:<5} n={v['n']:<5} "
                  f"dur={dm:.1f}s [{dmin:.0f}-{dmax:.0f}]  "
                  f"{w}x{h}@{fps:.1f}fps")
    print()
    for src, v in sorted(comp["by_label_source"].items(), key=lambda x: -x[1]["n"]):
        print(f"  label_source={src:<28} n={v['n']:<5}  conf={(v['confidence'] or 0):.3f}")

    # dump everything
    out = {
        "tab_gap_triple": t4,
        "per_detector_slice_auroc_tpr5": per_det_slice,
        "per_detector_platform_triple": per_det_plat,
        "m0_composition": comp,
    }
    out_path = RESULTS / "paper_numbers.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
