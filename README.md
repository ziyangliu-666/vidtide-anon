# VidTide — Anonymous Code Repository

> **Anonymous repository for the NeurIPS 2026 Datasets & Benchmarks Track double-blind submission**
> *"A Living In-the-Wild Benchmark for Measuring the Static-Benchmark Gap in AI-Generated Video Detection"*

This repository hosts the full pipeline source, the M0 slice manifest, and the
evaluation scripts that reproduce every number in the paper:

1. **Crawler pipeline** — platform-specific harvesters for AI-disclosure-labelled videos on YouTube, Bilibili, Reddit, and official model showcase galleries.
2. **Filter & dedup stack** — quality filter, label-tier filter, LLM-assisted semantic verification, and CLIP-based cross-platform deduplication.
3. **M0 slice manifest** — `manifests/M0/metadata.jsonl` + train / test / gap-eval splits. **Videos themselves are not redistributed**; each manifest entry carries the original platform URL and `scripts/download_videos.py` recovers the clip at evaluation time, following the VidProM precedent.
4. **Evaluation scripts** — linear-probe (LP) and full fine-tune (FT) trainers, Table 1 (GenVideo gap) reproducer, per-generator / per-platform breakdown tools, and the `extract_paper_numbers.py` driver that emits every number in the paper as JSON.
5. **Datasheet & reproducibility docs** — see `DATASHEET.md`, `docs/REPRODUCIBILITY.md`, and the Croissant 1.0 metadata file `croissant.json`.

> **M0 manifest count.** The shipped `manifests/M0/metadata.jsonl` contains
> **21,505** records (real 10,002 / fake 11,503), exported from the
> pre-submission database snapshot dated 2026-04-15. The paper datasheet
> reports a slightly larger raw M0 pool of **22,869** records (real 12,767 /
> fake 10,102) reflecting the full crawl window through to the submission
> cut-off; the ~6 % delta is dominated by Bilibili `real` clips that were
> hard-deleted by their uploaders between 2026-04-15 and submission and is
> documented in `manifests/M0/README.md`. **All headline experiments use
> the 9,956-clip class-balanced gap-evaluation slice
> (`manifests/M0/splits/gap_test.jsonl`), which is reproduced exactly here.**

---

## Repository layout

```
.
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── DATASHEET.md               ← NeurIPS-standard dataset documentation
├── CITATION.cff               ← anonymised
├── croissant.json             ← Croissant 1.0 dataset metadata
├── pyproject.toml             ← installable Python package (vidtide_anon)
├── requirements.txt           ← pip dependencies
│
├── crawlers/                  ← per-platform harvesters (re-exports server.crawler.*)
├── filters/                   ← BaseFilter implementations (re-exports server.filter.*)
├── dedup/                     ← CLIP-based cross-platform near-dup detector
├── pipeline/                  ← end-to-end pipeline runner
├── release/                   ← HuggingFace publishing stage
├── eval/                      ← detector adapters (DeMamba, NSG-VD, STIL, NPR, TALL)
│
├── server/                    ← canonical implementation (FastAPI app + library)
│   ├── crawler/               ← youtube.py, bilibili.py, bilibili_real.py,
│   │                            reddit.py, showcase.py, douyin.py, kuaishou.py,
│   │                            kinetics.py, pexels.py, registry.py
│   ├── filter/                ← quality_filter.py, tag_filter.py, llm_filter.py,
│   │                            model_whitelist_filter.py
│   ├── dedup/                 ← deduplicator.py, image_embedder.py,
│   │                            captioner.py, vec_index.py
│   ├── pipeline/              ← runner.py, postprocess.py
│   ├── release/               ← hf_publisher.py
│   ├── detection/             ← runner.py, dataset.py, metrics.py, registry.py,
│   │   └── detectors/           detectors/{demamba,nsgvd,stil,npr,tall}_*.py,
│   │                            clip_zero_shot.py, gpt4o_vision.py
│   ├── routers/               ← FastAPI endpoints (videos, slices, dedup,
│   │                            pipeline, review, stats, …)
│   ├── db/                    ← SQLAlchemy models + migrations
│   ├── storage/               ← blob storage layer
│   ├── utils/                 ← shared helpers
│   └── main.py                ← FastAPI app entry point
│
├── manifests/
│   └── M0/
│       ├── metadata.jsonl     ← 21,505 records (id, source_url, label,
│       │                        generator, tier_source, …)
│       ├── splits/
│       │   ├── train.jsonl    ← 7,957 LP/FT training records
│       │   ├── test.jsonl     ← 1,999 LP/FT test records
│       │   └── gap_test.jsonl ← 9,956 class-balanced gap-eval slice
│       │                        (paper Table 1)
│       └── README.md          ← schema, license, opt-out instructions
│
├── scripts/                   ← 40+ entry points; key ones below
│   ├── download_videos.py     ← recover clips from a manifest's source_url
│   ├── crawl_and_push.py      ← end-to-end pipeline driver
│   ├── compute_gap.py         ← Table 1 reproducer (static-benchmark gap)
│   ├── ft_train_lp.py         ← LP head training (Swin-T / TSM / I3D / SlowFast)
│   ├── ft_eval_all_slices.py  ← Table 2 reproducer (LP / FT recovery)
│   ├── ft_extract_features.py ← K400-init feature cache for LP heads
│   ├── ft_make_split.py       ← stratified train/test split builder
│   ├── nsgvd_train_discriminator.py
│   │                          ← NSG-VD Crafter checkpoint re-train
│   │                            (Appendix sec:appendix_nsgvd_crafter)
│   ├── nsgvd_extract_features.py / nsgvd_eval_best.py
│   ├── mmaction_*.py          ← MMAction2-based full-FT pipeline
│   ├── demamba_full_ft.py     ← DeMamba full fine-tune
│   ├── cleanlab_label_audit.py← Section 5 cleanlab analysis (93/100 flags)
│   ├── near_dup_audit.py      ← Section 5 pHash near-duplicate audit
│   ├── label_audit.py         ← runs both audits in sequence
│   ├── hard_case_analysis.py  ← Appendix hard-case stratification
│   ├── per_generator_full_pool.py
│   │                          ← per-generator AUROC over the wider 13,835 pool
│   ├── slice_bench_results.py ← Appendix per-platform / per-generator breakdown
│   ├── eval_lp_bilionly.py    ← Bilibili-only LP control (Appendix sec:appendix_bili)
│   ├── ood_eval_lp.py / download_ood_fresh.py
│   │                          ← Appendix M1-style OOD probe
│   ├── data_scaling_driver.py / plot_data_scaling.py
│   │                          ← Appendix data-scaling figure
│   ├── extract_paper_numbers.py
│   │                          ← dumps every paper table number as JSON
│   ├── plot_paper_figures.py  ← regenerates all paper figures from results/
│   ├── format_e1_results.py   ← LaTeX-ready formatter for Tables 1/2
│   ├── recompute_dedup.py     ← recompute CLIP dedup over the current DB
│   ├── tier2_review_tool.py   ← human-review CLI for tier-2 audit
│   ├── filter_vtuber_tutorial.py
│   │                          ← LLM filter for tutorial-not-AI clips
│   ├── predict_unknown_generators.py / cluster_unknown.py
│   │                          ← unknown-generator clustering (Appendix)
│   ├── analyze_predict_results.py / try_backbone_features.py
│   ├── backfill_published_at.py / age_videos.py / process_pending.py
│   │                          ← maintenance utilities
│   ├── run_pipeline.py / run_eval.py / bench_small.py
│   │                          ← shell-friendly drivers
│   └── download_nsgvd_crafter_weights.py
│                              ← fetches NSG-VD Crafter checkpoint we trained
│
├── config/
│   └── pipeline.yaml.example  ← crawler / filter / dedup configuration template
│
└── docs/
    ├── REPRODUCIBILITY.md     ← step-by-step recipe for every paper number
    ├── PIPELINE.md            ← architecture & tier-source taxonomy
    └── OPT_OUT.md             ← 24-hour clip-removal policy & request flow
```

## Quick start

```bash
git clone <this-repo>
cd vidtide-anon
pip install -r requirements.txt
pip install -e .            # registers vidtide_anon as an importable package

# 1. Recover the gap-evaluation slice from original platform URLs
python scripts/download_videos.py \
    --manifest manifests/M0/splits/gap_test.jsonl \
    --out data/M0/gap_test/

# 2. Reproduce the static-benchmark gap (paper Table 1)
python scripts/compute_gap.py \
    --bench-root data/M0/gap_test/ \
    --gap-test manifests/M0/splits/gap_test.jsonl \
    --detectors demamba_pika,demamba_crafter,nsgvd_pika,stil_pika,stil_crafter,npr_pika,tall_pika,clip_zero_shot \
    --out results/gap.csv

# 3. Reproduce the fine-tune recovery (paper Table 2)
python scripts/ft_extract_features.py --backbone swin_t --split train
python scripts/ft_extract_features.py --backbone swin_t --split test
python scripts/ft_train_lp.py        --backbone swin_t --epochs 20
python scripts/ft_eval_all_slices.py --backbone swin_t --eval-slice M0

# 4. Re-emit every paper-table number as JSON
python scripts/extract_paper_numbers.py
```

The first command is bandwidth-bound (≈80 GB at 720p+). For a smoke test,
add `--limit 200` to download only the first 200 clips of the gap split.

## Live leaderboard

The paper references a live leaderboard run by the authors. The instance URL
is left out of this repository to preserve double-blind anonymity; reviewers
can reproduce all numbers locally via the commands above without consulting
the live instance. The leaderboard URL will be added to the camera-ready
version.

## License

MIT — see `LICENSE`. The released **manifest** (metadata only, no video files)
follows the **CC-BY-NC 4.0** precedent set by VidProM; per-clip rights remain
with the original uploaders. See `manifests/M0/README.md` for the schema and
opt-out workflow.

## Reproducibility & Datasheet

- `DATASHEET.md` — full NeurIPS Datasets & Benchmarks–style dataset card (motivation, composition, collection, preprocessing, uses, distribution, maintenance).
- `croissant.json` — Croissant 1.0 machine-readable dataset metadata (NeurIPS RAI requirement).
- `docs/REPRODUCIBILITY.md` — exact command sequence to reproduce every table and figure in the paper.
- `docs/PIPELINE.md` — architecture diagram, tier-source taxonomy, cost breakdown.
- `docs/OPT_OUT.md` — 24-hour opt-out policy and the (anonymised) issue-tracker workflow.
