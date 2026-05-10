"""Render the VIDTIDE data-scaling figure as a dense 2x3 panel.
import os

Panels (left-to-right, top-to-bottom):
  (a) Scaling curves: test AUROC vs #training videos (log x), per backbone,
      with GenVideo-trained baseline horizontal references (where they exist).
  (b) Marginal gain per doubling: grouped bars of Delta AUROC across the
      three fraction transitions, per backbone. Quantifies saturation.
  (c) Test AUROC vs epoch for DeMamba at all 4 fractions.
  (d) Train-loss trajectories for DeMamba at all 4 fractions (log y).
  (e) Compute vs AUROC: per-run wall-clock seconds (log x) vs final AUROC,
      per backbone, with fraction encoded by marker size.
  (f) Peak-AUROC epoch per (backbone, fraction): shows how long each run
      keeps improving before plateauing.

Reads results/data_scaling.json; writes figs/data_scaling.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PAPER_FIGS = Path(os.environ.get("PAPER_FIGS_DIR", "figs"))

BACKBONES = ["tsm", "demamba", "swin"]
COLORS = {"tsm": "#1f77b4", "demamba": "#d62728", "swin": "#2ca02c"}
LABELS = {"tsm": "TSM", "demamba": "DeMamba", "swin": "VideoSwin-T"}
MARKERS = {"tsm": "o", "demamba": "s", "swin": "^"}
FRACS = [0.10, 0.25, 0.50, 1.00]
FRAC_KEYS = ["0.10", "0.25", "0.50", "1.00"]
FRAC_LABELS = {0.10: "10%", 0.25: "25%", 0.50: "50%", 1.00: "100%"}
# Fraction -> viridis-ish shade for panels (c)/(d)
FRAC_CMAP = plt.get_cmap("viridis")
FRAC_COLORS = {f: FRAC_CMAP(i / (len(FRACS) - 1) * 0.85) for i, f in enumerate(FRACS)}

# GenVideo-trained Original-checkpoint baselines (from Table 8 / Table 4).
# Only DeMamba has a public GenVideo checkpoint among our three LP backbones.
GENVIDEO_BASELINES = {"demamba": 0.694}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO / "results" / "data_scaling.json"))
    ap.add_argument("--out-dir", default=str(PAPER_FIGS))
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slightly taller figure + bottom row reserved for a shared legend strip
    # so that per-panel legends never have to sit on top of the data.
    fig = plt.figure(figsize=(6.9, 4.6))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.0, 0.12],
        hspace=0.62, wspace=0.42,
        left=0.07, right=0.985, top=0.94, bottom=0.04,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        ax.tick_params(labelsize=7)

    # -------- (a) AUROC vs #training videos ----------------------------------
    for bb in BACKBONES:
        if bb not in data:
            continue
        rows = sorted(data[bb].items(), key=lambda kv: float(kv[0]))
        xs = [v["n_used"] for _, v in rows]
        ys = [v["best_auroc"] for _, v in rows]
        ax_a.plot(xs, ys, color=COLORS[bb], marker=MARKERS[bb], linewidth=1.6,
                  markersize=5, label=LABELS[bb])
    # GenVideo baseline: short dotted segment on the left side only, labeled
    # immediately above, far below the rising curves — no collision.
    for bb, y0 in GENVIDEO_BASELINES.items():
        ax_a.plot([700, 1400], [y0, y0], color=COLORS[bb],
                  linestyle=":", linewidth=1.1, alpha=0.85)
        ax_a.annotate(f"GenVideo ckpt {y0:.2f}",
                      xy=(1400, y0), xytext=(1600, y0),
                      textcoords="data", color=COLORS[bb], fontsize=6.2,
                      va="center", ha="left")
    ax_a.set_xscale("log")
    ax_a.set_xticks([800, 2000, 4000, 8000])
    ax_a.set_xticklabels(["0.8K", "2K", "4K", "8K"])
    ax_a.set_xlabel("Training videos", fontsize=8)
    ax_a.set_ylabel("Test AUROC", fontsize=8)
    ax_a.set_ylim(0.66, 0.98)
    ax_a.grid(True, which="both", linestyle=":", alpha=0.45)
    ax_a.set_title("(a) Scaling curves", fontsize=8.5, loc="left")

    # -------- (b) Marginal gain per doubling ---------------------------------
    trans = [("0.10", "0.25"), ("0.25", "0.50"), ("0.50", "1.00")]
    trans_labels = [r"0.8K$\!\to\!$2K", r"2K$\!\to\!$4K", r"4K$\!\to\!$8K"]
    x = np.arange(len(trans)) * 1.25  # extra space between groups
    width = 0.32
    for i, bb in enumerate(BACKBONES):
        if bb not in data:
            continue
        deltas = [100.0 * (data[bb][b]["best_auroc"] - data[bb][a]["best_auroc"])
                  for a, b in trans]
        bars = ax_b.bar(x + (i - 1) * width, deltas, width,
                        color=COLORS[bb], edgecolor="white", linewidth=0.6)
        for rect, d in zip(bars, deltas):
            ax_b.text(rect.get_x() + rect.get_width() / 2,
                      rect.get_height() + 0.08, f"{d:.1f}",
                      ha="center", va="bottom", fontsize=6.2, color=COLORS[bb])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(trans_labels, fontsize=7)
    ax_b.set_ylabel(r"$\Delta$ AUROC (pp)", fontsize=8)
    ax_b.set_ylim(0, 3.8)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_b.set_title("(b) Marginal gain / doubling", fontsize=8.5, loc="left")

    # -------- (c) Test AUROC vs epoch (DeMamba, 4 fractions) -----------------
    for f, k in zip(FRACS, FRAC_KEYS):
        pe = data["demamba"][k]["per_epoch"]
        xs = [e["epoch"] for e in pe]
        ys = [e["test_auroc"] for e in pe]
        ax_c.plot(xs, ys, color=FRAC_COLORS[f], linewidth=1.3,
                  label=FRAC_LABELS[f])
    ax_c.set_xlabel("Epoch (DeMamba)", fontsize=8)
    ax_c.set_ylabel("Test AUROC", fontsize=8)
    ax_c.set_xlim(0, 52)
    ax_c.set_ylim(0.55, 0.98)
    ax_c.grid(True, linestyle=":", alpha=0.45)
    ax_c.set_title("(c) Convergence (test)", fontsize=8.5, loc="left")

    # -------- (d) Train-loss trajectories ------------------------------------
    for f, k in zip(FRACS, FRAC_KEYS):
        pe = data["demamba"][k]["per_epoch"]
        xs = [e["epoch"] for e in pe]
        ys = [max(e["train_loss"], 1e-4) for e in pe]
        ax_d.plot(xs, ys, color=FRAC_COLORS[f], linewidth=1.3)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Epoch (DeMamba)", fontsize=8)
    ax_d.set_ylabel("Train loss (log)", fontsize=8)
    ax_d.grid(True, which="both", linestyle=":", alpha=0.45)
    ax_d.set_title("(d) Training loss", fontsize=8.5, loc="left")

    # -------- (e) Compute vs AUROC efficiency frontier -----------------------
    size_map = {0.10: 22, 0.25: 42, 0.50: 72, 1.00: 118}
    for bb in BACKBONES:
        if bb not in data:
            continue
        rows = sorted(data[bb].items(), key=lambda kv: float(kv[0]))
        xs = [max(v["elapsed_s"], 1) for _, v in rows]
        ys = [v["best_auroc"] for _, v in rows]
        sizes = [size_map[float(k)] for k, _ in rows]
        ax_e.plot(xs, ys, color=COLORS[bb], linewidth=1.0, alpha=0.55, zorder=1)
        ax_e.scatter(xs, ys, s=sizes, color=COLORS[bb], marker=MARKERS[bb],
                     edgecolor="white", linewidth=0.6, zorder=2)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("Training time (s, log)", fontsize=8)
    ax_e.set_ylabel("Test AUROC", fontsize=8)
    ax_e.set_xlim(2, 900)
    ax_e.set_ylim(0.82, 0.965)
    ax_e.grid(True, which="both", linestyle=":", alpha=0.45)
    ax_e.set_title("(e) Compute vs AUROC", fontsize=8.5, loc="left")
    # Inline note: marker size encodes training fraction
    ax_e.text(0.03, 0.96, "marker size $\\propto$ train frac.",
              transform=ax_e.transAxes, fontsize=6.2, color="#444",
              va="top", ha="left")

    # -------- (f) Gap closed over GenVideo ckpt ------------------------------
    bb = "demamba"
    if bb in data:
        rows = sorted(data[bb].items(), key=lambda kv: float(kv[0]))
        xs = np.arange(len(rows))
        base = GENVIDEO_BASELINES[bb]
        ys_abs = [v["best_auroc"] for _, v in rows]
        ys_gain = [(y - base) * 100.0 for y in ys_abs]
        bars = ax_f.bar(xs, ys_gain, color=COLORS[bb], edgecolor="white",
                        linewidth=0.6, width=0.6)
        for rect, g in zip(bars, ys_gain):
            ax_f.text(rect.get_x() + rect.get_width() / 2,
                      rect.get_height() + 0.5, f"+{g:.1f}",
                      ha="center", va="bottom", fontsize=6.5,
                      color=COLORS[bb])
        ax_f.axhline(0, color="gray", linewidth=0.8)
        ax_f.set_xticks(xs)
        ax_f.set_xticklabels([FRAC_LABELS[float(k)] for k, _ in rows], fontsize=7)
        ax_f.set_xlabel("Train frac. (DeMamba)", fontsize=8)
        ax_f.set_ylabel(r"$\Delta$ AUROC vs GenVideo (pp)", fontsize=8)
        ax_f.set_ylim(0, 34)
        ax_f.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax_f.set_title(f"(f) Gap closed (DeMamba, vs GenVideo {base:.2f})",
                       fontsize=8.5, loc="left")

    # ---- Shared legend strip (backbone colors + fraction shades) -----------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    backbone_handles = [
        Line2D([0], [0], color=COLORS[bb], marker=MARKERS[bb], linestyle="-",
               markersize=5, markeredgecolor="white", markeredgewidth=0.5,
               label=LABELS[bb])
        for bb in BACKBONES
    ]
    frac_handles = [
        Patch(facecolor=FRAC_COLORS[f], edgecolor="white",
              label=f"train frac. {FRAC_LABELS[f]}")
        for f in FRACS
    ]
    leg = legend_ax.legend(
        handles=backbone_handles + frac_handles,
        loc="center", frameon=False, ncol=7, fontsize=7.0,
        handletextpad=0.5, columnspacing=1.4, handlelength=1.6,
    )
    for txt in leg.get_texts():
        txt.set_fontsize(7.0)
    pdf_path = out_dir / "data_scaling.pdf"
    png_path = out_dir / "data_scaling.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    print(f"→ {pdf_path}\n→ {png_path}")


if __name__ == "__main__":
    main()
