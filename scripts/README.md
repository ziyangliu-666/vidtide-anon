# Scripts

Top-level entry points used to reproduce paper numbers.

| Script | Reproduces | Notes |
|--------|------------|-------|
| `download_videos.py` | Recovers clips from a manifest's `source_url` field | Validates `sha256` where possible; logs platform-re-encoding mismatches |
| `crawl_and_push.py` | End-to-end pipeline: crawl → filter → dedup → audit → publish | Single-shot driver |
| `compute_gap.py` | Table 1 — static-benchmark gap (GenVideo AUROC vs. VidTide M0 AUROC) | Per-detector and pooled |
| `ft_train_lp.py` | Linear-probe head training (Swin-T / TSM / I3D / SlowFast on K400-init features) | Hyperparameters in Appendix `tab:appendix_hparams` |
| `ft_eval_all_slices.py` | Table 2 — LP / FT evaluation across slices | M0 in-distribution + (in camera-ready) M1 temporal generalisation |
| `label_audit.py` | Cleanlab-based label-quality audit ($\kappa = 0.93$) | |
| `compute_metrics.py` | AUROC / bACC / F1 utilities (threshold 0.5 for bACC and F1) | Shared by the above |

> **Skeleton notice.** Stub scripts and a `requirements.txt` land in the next batch. See `docs/REPRODUCIBILITY.md` for the planned end-to-end recipe.
