# VidTide — Anonymous Code Repository

> **Anonymous repository for the NeurIPS 2026 Datasets & Benchmarks Track double-blind submission**
> *"A Living In-the-Wild Benchmark for Measuring the Static-Benchmark Gap in AI-Generated Video Detection"*

This repository accompanies the submission and will host:

1. **Crawler pipeline** — platform-specific harvesters for AI-disclosure-labelled videos on YouTube, Bilibili, Reddit, and official model showcase galleries.
2. **Filter & dedup stack** — quality filter, label-tier filter, LLM-assisted semantic verification, and CLIP-based cross-platform deduplication.
3. **Monthly slice manifests** — `metadata.jsonl` + train/val/test splits for each frozen benchmark slice (M0, M1, ...). **Videos themselves are not redistributed**; each manifest entry carries the original platform URL and a download script recovers the clip at evaluation time, following the VidProM precedent.
4. **Evaluation scripts** — linear-probe (LP) and full fine-tune (FT) trainers, Table 1 (GenVideo gap) reproducer, and per-generator / per-platform breakdown tools.
5. **Datasheet & reproducibility docs** — see `DATASHEET.md` and `docs/REPRODUCIBILITY.md`.

> **Status (submission-time skeleton).** This anonymous mirror currently contains the directory structure, the datasheet, the reproducibility plan, and stubs that document each module's interface. The full pipeline source, the `M0` manifest (22,869 clips), and the evaluation scripts that produce every number in the paper are **uploaded in batches as the double-blind review window opens**, to keep the anonymisation review of each file tractable. All paper numbers are computed by the scripts in `scripts/` and `eval/`; the corresponding files are linked from each section below.

---

## Repository layout

```
.
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── DATASHEET.md               ← NeurIPS-standard dataset documentation
├── CITATION.cff               ← anonymised
│
├── crawlers/                  ← per-platform harvesters (BaseCrawler interface)
│   ├── youtube.py             ← yt-dlp search + AI-disclosure tag detection
│   ├── bilibili.py            ← yt-dlp bilisearch + Chinese AI-keyword + argue_info tag
│   ├── reddit.py              ← Reddit JSON API (r/aivideo, r/sora, ...)
│   └── showcase.py            ← official-gallery scrapers (Pika, Kling, Runway, ...)
│
├── filters/                   ← BaseFilter interface
│   ├── quality_filter.py      ← resolution / duration / fps thresholds
│   ├── tag_filter.py          ← T1 / T2 / T3 label-tier gating
│   └── llm_filter.py          ← GPT-4o title/description verification
│
├── dedup/
│   └── clip_dedup.py          ← CLIP image-embedder + cross-platform near-dup index
│
├── manifests/                 ← frozen monthly slices (metadata + split files only)
│   └── M0/
│       ├── metadata.jsonl     ← 22,869 records (id, source_url, label, generator, ...)
│       ├── splits/
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   └── test.jsonl     ← 5K real / 5K fake gap-evaluation slice
│       └── README.md          ← schema, license, opt-out instructions
│
├── scripts/
│   ├── download_videos.py     ← recovers clips from source_url at eval time
│   ├── crawl_and_push.py      ← end-to-end pipeline driver
│   ├── compute_gap.py         ← Table 1 (static-benchmark gap) reproducer
│   ├── ft_train_lp.py         ← linear-probe training (Swin-T / TSM / I3D / SlowFast)
│   ├── ft_eval_all_slices.py  ← LP / FT evaluation across slices
│   └── label_audit.py         ← Cleanlab-based label-quality audit
│
├── eval/                      ← detector wrappers used in Table 1
│   └── README.md              ← DeMamba, NSG-VD, STIL, NPR, TALL adapter notes
│
├── config/
│   └── pipeline.yaml.example  ← crawler / filter / dedup configuration template
│
└── docs/
    ├── REPRODUCIBILITY.md     ← step-by-step recipe for every paper number
    ├── PIPELINE.md            ← architecture & tier-source taxonomy
    └── OPT_OUT.md             ← 24-hour clip-removal policy & request flow
```

## Quick start (after the full release lands)

```bash
git clone <this-repo>
cd vidtide-anon
pip install -r requirements.txt

# 1. Recover the M0 evaluation slice from original platform URLs
python scripts/download_videos.py \
    --manifest manifests/M0/splits/test.jsonl \
    --out data/M0/test/

# 2. Reproduce the static-benchmark gap (Table 1)
python scripts/compute_gap.py \
    --bench-root data/M0/test/ \
    --detectors demamba_pika,demamba_crafter,nsgvd_pika,stil_pika,npr_pika,tall_pika \
    --out results/gap.csv

# 3. Reproduce the fine-tune recovery (Table 2)
python scripts/ft_train_lp.py --backbone swin_t --slice M0 --epochs 20
python scripts/ft_eval_all_slices.py --backbone swin_t --eval-slice M0
```

## License

MIT — see `LICENSE`. The released **manifests** (metadata only, no video files) follow the **CC-BY-NC 4.0** precedent set by VidProM; per-clip rights remain with the original uploaders. See `manifests/M0/README.md`.

## Reproducibility & Datasheet

- `DATASHEET.md` — full NeurIPS Datasets & Benchmarks–style dataset card (motivation, composition, collection, preprocessing, uses, distribution, maintenance).
- `docs/REPRODUCIBILITY.md` — exact command sequence to reproduce every table and figure in the paper.
- `docs/PIPELINE.md` — architecture diagram, tier-source taxonomy, cost breakdown.
- `docs/OPT_OUT.md` — 24-hour opt-out policy and the (anonymised) issue-tracker workflow.

## Live leaderboard

The live leaderboard URL referenced in the paper is **anonymised** for the double-blind submission window. Reviewers can run all evaluations locally via this repository; the live web instance will be linked in the camera-ready version.
