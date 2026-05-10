# Scripts

Top-level entry points used to reproduce paper numbers, plus pipeline,
crawl, and audit utilities. Forty-plus scripts; the most important ones
are listed in the top-level [`README.md`](../README.md#repository-layout).

`scripts/extract_paper_numbers.py` is the canonical "every paper number
as JSON" driver; `scripts/plot_paper_figures.py` regenerates every paper
figure from cached results. Together they are sufficient to reproduce all
tables and plots assuming the cached `results/bench_5k_*_scores.jsonl`
score files are present (re-create them with `scripts/compute_gap.py`).

A typical end-to-end reproduction run:

```bash
# 1. Recover videos
python scripts/download_videos.py --manifest manifests/M0/splits/gap_test.jsonl \
    --out data/M0/gap_test/

# 2. Score the seven detectors over the gap slice (~24 h on one A100)
python scripts/compute_gap.py --bench-root data/M0/gap_test/ \
    --detectors demamba_pika,demamba_crafter,nsgvd_pika,stil_pika,stil_crafter,npr_pika,tall_pika,clip_zero_shot

# 3. Train the LP heads + evaluate (Table 2)
for bk in swin_t tsm i3d slowfast; do
    python scripts/ft_extract_features.py --backbone $bk --split train
    python scripts/ft_extract_features.py --backbone $bk --split test
    python scripts/ft_train_lp.py        --backbone $bk --epochs 20
    python scripts/ft_eval_all_slices.py --backbone $bk --eval-slice M0
done

# 4. Run label / dedup audits (Section 5)
python scripts/label_audit.py

# 5. Emit every paper-table number
python scripts/extract_paper_numbers.py
python scripts/plot_paper_figures.py
```

See `docs/REPRODUCIBILITY.md` for the per-table walkthrough.
