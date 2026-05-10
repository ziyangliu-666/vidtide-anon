"""Render paper figures from cached results JSONs.
import os

Outputs PDF + PNG into figs/ (committed to paper repo).

Figure mapping:
  Fig 4a — per-platform heatmap (detectors × platforms)
  Fig 4b — per-generator heatmap (detectors × generators)
  Fig 6  — hard-case mosaic (top-K hardest fake videos × 4 sample frames)
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("figs")

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
PAPER_FIGS = Path(os.environ.get("PAPER_FIGS_DIR", "figs"))
BLOB_ROOT = REPO / "data" / "blobs" / "videos"

DETECTOR_DISPLAY = {
    "npr_pika": "NPR (Pika)", "npr_crafter": "NPR (Crafter)",
    "tall_pika": "TALL (Pika)", "tall_crafter": "TALL (Crafter)",
    "stil_pika": "STIL (Pika)", "stil_crafter": "STIL (Crafter)",
    "demamba_pika": "DeMamba (Pika)", "demamba_crafter": "DeMamba (Crafter)",
    "nsgvd_pika": "NSG-VD (Pika)", "nsgvd_crafter": "NSG-VD (Crafter)",
    "clip_zero_shot": "CLIP (zero-shot)",
}
PLAT_DISPLAY = {"bilibili": "Bilibili", "reddit": "Reddit",
                "youtube": "YouTube", "showcase": "Showcase"}


def _heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
             title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.7 * len(col_labels) + 2, 0.4 * len(row_labels) + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.55 < v < 0.85 else "white", fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="AUROC")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("→ %s.{pdf,png}", out_path)


def fig_per_platform() -> None:
    d = json.loads((RESULTS / "per_platform_auroc.json").read_text())["results"]
    detectors = sorted(d.keys())
    platforms: list[str] = []
    for det in detectors:
        for p in d[det].keys():
            if p not in platforms: platforms.append(p)
    platforms = sorted(platforms)

    matrix = np.full((len(detectors), len(platforms)), np.nan, dtype=float)
    for i, det in enumerate(detectors):
        for j, plat in enumerate(platforms):
            row = d[det].get(plat)
            if row: matrix[i, j] = row.get("auroc", np.nan)

    _heatmap(
        matrix,
        [DETECTOR_DISPLAY.get(d_, d_) for d_ in detectors],
        [PLAT_DISPLAY.get(p, p) for p in platforms],
        "Per-platform AUROC",
        PAPER_FIGS / "heatmap_platform",
    )


def fig_per_generator() -> None:
    d = json.loads((RESULTS / "per_generator_auroc.json").read_text())["results"]
    detectors = sorted(d.keys())
    gen_set: set[str] = set()
    for det in detectors:
        gen_set.update(d[det].keys())
    # Drop _unlabeled (no value for paper figure) and sort by total fake count
    counts = {g: max((d[det].get(g, {}).get("n_fake", 0)) for det in detectors)
              for g in gen_set if g != "_unlabeled"}
    generators = sorted(counts.keys(), key=lambda g: -counts[g])

    matrix = np.full((len(detectors), len(generators)), np.nan, dtype=float)
    for i, det in enumerate(detectors):
        for j, gen in enumerate(generators):
            row = d[det].get(gen)
            if row: matrix[i, j] = row.get("auroc", np.nan)

    _heatmap(
        matrix,
        [DETECTOR_DISPLAY.get(d_, d_) for d_ in detectors],
        generators,
        "Per-generator AUROC",
        PAPER_FIGS / "heatmap_generator",
    )


def _grab_frame(video: Path, t_frac: float = 0.5) -> Image.Image | None:
    """Use ffmpeg to grab a single frame at fractional time, return PIL.Image."""
    if not video.exists():
        return None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, timeout=20,
    )
    try:
        dur = float(probe.stdout.strip())
    except Exception:
        return None
    t = max(0.1, dur * t_frac)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "f.jpg"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{t:.2f}", "-i", str(video),
               "-frames:v", "1", "-vf", "scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
               str(out)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=20, check=True)
        except Exception:
            return None
        return Image.open(out).convert("RGB").copy()


def fig_hard_cases(top_n: int = 20) -> None:
    d = json.loads((RESULTS / "hard_cases.json").read_text())
    rows = d["top_hard_fakes"][:top_n]

    # Layout: 4 cols × ceil(top_n/4) rows
    cols = 4
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.2, n_rows * 2.4))
    axes = np.array(axes).reshape(n_rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for i, row in enumerate(rows):
        r, c = divmod(i, cols)
        vid = row["video"]
        img = _grab_frame(BLOB_ROOT / "fake" / vid, 0.5)
        if img is None:
            continue
        axes[r, c].imshow(img)
        axes[r, c].set_title(
            f"{row.get('platform','?')}/{row.get('generator') or 'unlabeled'}\n"
            f"ensemble AI-prob={row['ensemble_ai_prob']:.2f}",
            fontsize=7, pad=2,
        )
        axes[r, c].axis("off")

    fig.suptitle(f"Hardest-to-detect AI clips (top {len(rows)})", fontsize=11)
    fig.tight_layout()
    out = PAPER_FIGS / "hard_cases"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("→ %s.{pdf,png}", out)


def main() -> None:
    fig_per_platform()
    fig_per_generator()
    fig_hard_cases(top_n=20)
    log.info("done")


if __name__ == "__main__":
    main()
