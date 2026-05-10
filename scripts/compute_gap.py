"""Reproduce paper Table 1 (static-benchmark gap).

Runs each requested detector over the gap-evaluation slice
(``manifests/M0/splits/gap_test.jsonl``, 9,956 clips) and prints
AUROC / bACC / F1 per detector along with the published GenVideo
in-domain AUROC and the resulting gap (paper column ``Δ``).

Usage
-----

    python scripts/compute_gap.py \
        --bench-root data/M0/test/ \
        --detectors demamba_pika,demamba_crafter,nsgvd_pika,stil_pika,npr_pika,tall_pika \
        --gap-test manifests/M0/splits/gap_test.jsonl \
        --out results/gap.csv

The detector adapters live under ``server/detection/detectors/``; each
fetches its upstream checkpoint from the original authors' GitHub release
(we never redistribute upstream weights).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Published GenVideo AUROC for each detector × checkpoint, taken from the
# corresponding original papers (paper Table 1, "GenVideo" column).
GENVIDEO_AUROC = {
    "demamba_pika":      0.9613,
    "demamba_crafter":   0.9466,
    "stil_pika":         0.9412,
    "stil_crafter":      0.9344,
    "npr_pika":          0.9345,
    "npr_crafter":       0.9347,
    "tall_pika":         0.9540,
    "tall_crafter":      0.9540,
    "nsgvd_pika":        0.9420,
    "nsgvd_crafter":     0.9420,  # re-trained by us; see appendix sec:appendix_nsgvd_crafter
    "clip_zero_shot":    None,    # no in-domain reference
}


def load_gap_test(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", required=True, type=Path,
                    help="Directory containing the downloaded gap_test mp4s "
                         "(one file per id, named {id}.mp4).")
    ap.add_argument("--detectors", required=True,
                    help="Comma-separated detector names matching modules under "
                         "server/detection/detectors/.")
    ap.add_argument("--gap-test", type=Path,
                    default=Path("manifests/M0/splits/gap_test.jsonl"),
                    help="Path to the gap-evaluation manifest (9,956 clips).")
    ap.add_argument("--out", type=Path, default=Path("results/gap.csv"),
                    help="Output CSV path.")
    args = ap.parse_args()

    # Lazy imports so the script can at least print --help without the full
    # detector dependency tree installed.
    from server.detection.registry import get_detector  # noqa: E402
    from server.detection.runner import score_videos    # noqa: E402
    from server.detection.metrics import auroc, bacc, f1  # noqa: E402

    gap = load_gap_test(args.gap_test)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for det_name in detectors:
        print(f"=== {det_name} ===", file=sys.stderr)
        detector = get_detector(det_name)
        scores = score_videos(detector, gap, args.bench_root)
        labels = [1 if gap[vid]["label"] == "fake" else 0 for vid in scores]
        preds  = [scores[vid] for vid in scores]
        auc = auroc(labels, preds)
        ba  = bacc(labels, preds, threshold=0.5)
        f   = f1(labels, preds, threshold=0.5)
        gv  = GENVIDEO_AUROC.get(det_name)
        delta = (auc - gv) if gv is not None else None
        print(f"  AUROC={auc*100:.1f}  bACC={ba*100:.1f}  F1={f*100:.1f}  "
              f"GenVideo={gv*100 if gv else float('nan'):.1f}  Δ={delta*100 if delta is not None else float('nan'):+.1f}",
              file=sys.stderr)
        rows.append({
            "detector": det_name,
            "vidtide_auroc": round(auc, 4),
            "vidtide_bacc":  round(ba, 4),
            "vidtide_f1":    round(f, 4),
            "genvideo_auroc": gv,
            "delta_auroc": round(delta, 4) if delta is not None else None,
            "n_eval": len(scores),
        })

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
