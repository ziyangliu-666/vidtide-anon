"""Format E1 (VidTide LP → GenVideo) results for paper.

Reads results/vidtide_to_genvideo.json (downloaded from Modal volume after
the pipeline completes) and emits:
  1. A LaTeX-ready per-(backbone, generator) AUROC table
  2. Pooled AUROC with WildScrape correctly excluded (it's a GenVideo training
     real, not a fake — the on-disk path puts it under fake/ which our scan
     mislabeled).

Usage:
    modal volume get vidtide-genvideo-eval results.json results/vidtide_to_genvideo.json
    python scripts/format_e1_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
INPUT = RESULTS / "vidtide_to_genvideo.json"

# Per-backbone feature .npz files we recompute pooled AUROC over.
# Note: WildScrape is GenVideo's *real* training class, but the directory
# scanner placed it under fake/ — flag it for re-classification.
WILDSCRAPE_KEY = "fake_WildScrape"
REAL_KEY_HINTS = ("real_msr-vtt", "real_MSR-VTT", "real_msr_vtt", "msr-vtt", "msr_vtt")


def main():
    if not INPUT.exists():
        print(f"missing {INPUT}", file=sys.stderr)
        print("did you run: modal volume get vidtide-genvideo-eval results.json results/vidtide_to_genvideo.json ?",
              file=sys.stderr)
        sys.exit(1)
    R = json.loads(INPUT.read_text())

    # Generator display names for paper ordering
    DISPLAY_ORDER = [
        ("fake_Crafter", "Crafter"),
        ("fake_Lavie", "Lavie"),
        ("fake_Gen2", "Gen2"),
        ("fake_HotShot", "HotShot-XL"),
        ("fake_ModelScope", "ModelScope"),
        ("fake_MorphStudio", "MorphStudio"),
        ("fake_Show_1", "Show-1"),
        ("fake_MoonValley", "MoonValley"),
        ("fake_Sora", "Sora v1"),
    ]

    backbones = ["swin", "tsm", "i3d", "slowfast"]

    # --- Section 1: per-(backbone, generator) AUROC table ---
    print("=" * 80)
    print("Per-(backbone, generator) AUROC on GenVideo-Val (vs MSR-VTT real pool)")
    print("=" * 80)
    header = f"{'Generator':<14}"
    for bb in backbones:
        header += f" {bb.upper():>8}"
    header += f"  {'Avg':>6}"
    print(header)
    print("-" * len(header))

    rows_data = []  # for LaTeX
    for key, disp in DISPLAY_ORDER:
        cells = []
        nfakes = []
        nreals = []
        for bb in backbones:
            d = R.get(bb, {}).get("per_generator", {}).get(key)
            if d is None:
                cells.append(None)
                continue
            cells.append(d.get("auroc"))
            nfakes.append(d.get("n_fake"))
            nreals.append(d.get("n_real"))
        valid = [c for c in cells if c is not None]
        avg = np.mean(valid) if valid else None
        n_fake = max(nfakes) if nfakes else None
        n_real = max(nreals) if nreals else None
        row_str = f"{disp:<14}"
        for c in cells:
            row_str += f" {('  --  ' if c is None else f'{c*100:>6.2f}'):>8}"
        row_str += f"  {('--' if avg is None else f'{avg*100:>6.2f}'):>6}"
        row_str += f"  (n_f={n_fake}, n_r={n_real})"
        print(row_str)
        rows_data.append((disp, cells, avg, n_fake, n_real))

    # --- Section 2: WildScrape sanity (should be ~0.5 AUROC vs MSR-VTT) ---
    print()
    print("=" * 80)
    print("Sanity check: WildScrape (GenVideo training real, mislabeled fake by scan)")
    print("=" * 80)
    ws_cells = []
    for bb in backbones:
        d = R.get(bb, {}).get("per_generator", {}).get(WILDSCRAPE_KEY)
        ws_cells.append(d.get("auroc") if d else None)
        n_fake = d.get("n_fake") if d else 0
        n_real = d.get("n_real") if d else 0
        print(f"  {bb}: AUROC={d.get('auroc') if d else None}  (n={n_fake} 'wild scrape' vs n={n_real} MSR-VTT)")
    ws_valid = [c for c in ws_cells if c is not None]
    if ws_valid:
        print(f"  → WildScrape vs MSR-VTT mean AUROC: {np.mean(ws_valid)*100:.2f}%")
        print(f"  (interpretation: a value near 0.5 confirms LP heads do NOT use platform identity)")

    # --- Section 3: Pooled AUROC (per backbone, all real fakes vs MSR-VTT) ---
    print()
    print("=" * 80)
    print("Pooled cross-dataset AUROC (per backbone): all true-fake generators vs MSR-VTT")
    print("=" * 80)
    pooled = {}
    for bb in backbones:
        v = R.get(bb, {}).get("pooled_auroc")
        pooled[bb] = v
        print(f"  {bb}: pooled AUROC = {v*100:.2f}%" if v else f"  {bb}: pooled AUROC = N/A")
    valid = [v for v in pooled.values() if v is not None]
    if valid:
        print(f"  → 4-backbone average pooled AUROC: {np.mean(valid)*100:.2f}%")

    # --- Section 4: emit LaTeX rows for the appendix table ---
    print()
    print("=" * 80)
    print("LaTeX rows for tab:cross_dataset (paste into 9appendix.tex)")
    print("=" * 80)
    for disp, cells, avg, n_f, n_r in rows_data:
        latex_cells = []
        for c in cells:
            latex_cells.append("--" if c is None else f"{c*100:.1f}")
        avg_s = "--" if avg is None else f"{avg*100:.1f}"
        print(f"{disp:<14} & " + " & ".join(latex_cells) + f" & {avg_s} \\\\")
    # WildScrape line
    if ws_valid:
        ws_strs = [f"{c*100:.1f}" if c is not None else "--" for c in ws_cells]
        ws_avg = np.mean(ws_valid) * 100
        print()
        print("WildScrape (sanity, excluded from pool) & " + " & ".join(ws_strs) + f" & {ws_avg:.1f} \\\\")
    # Pooled
    pooled_strs = [f"{pooled[bb]*100:.1f}" if pooled[bb] is not None else "--" for bb in backbones]
    pooled_avg = np.mean([pooled[bb] for bb in backbones if pooled[bb] is not None]) * 100
    print()
    print("\\textbf{Pooled (all fakes vs.\\ MSR-VTT)} & " +
          " & ".join([f"\\textbf{{{s}}}" for s in pooled_strs]) +
          f" & \\textbf{{{pooled_avg:.1f}}} \\\\")


if __name__ == "__main__":
    main()
