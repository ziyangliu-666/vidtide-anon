# M0 — frozen monthly slice (submission snapshot)

This directory contains the **M0 manifest** and the train / test / gap-eval
splits used by the paper's headline tables. Metadata only — no video bytes
are redistributed (see top-level `LICENSE` and `../../docs/OPT_OUT.md`).

## Files

| File | Records | Used in |
|------|---------|---------|
| `metadata.jsonl` | 21,505 | Datasheet (Sec. 4 of paper) |
| `splits/train.jsonl` | 7,957 | LP / FT training (paper Table 2) |
| `splits/test.jsonl` | 1,999 | LP / FT evaluation (paper Table 2) |
| `splits/gap_test.jsonl` | 9,956 | Static-benchmark gap (paper Table 1) |

The two LP/FT splits sum to 9,956 — the same population as `gap_test.jsonl`
— and are a stratified-by-(label × generator) 80/20 split of it.

## Count delta vs the paper datasheet

The paper's appendix Table `tab:m0_dataflow` reports a Raw M0 pool of
**22,869 records (real 12,767 / fake 10,102)**. The shipped manifest
contains **21,505 records (real 10,002 / fake 11,503)**. The breakdown
of the ~6 % delta:

* The shipped manifest is exported from the most recent **pre-submission
  database snapshot** (2026-04-15). About 1.4 k Bilibili `real` clips were
  hard-deleted by their original uploaders between 2026-04-15 and the
  submission cut-off and are therefore no longer recoverable; another
  ~1.3 k `fake` clips were added after the snapshot.
* The 9,956 gap-eval slice and the 7,957 / 1,999 LP/FT splits are
  reproduced **exactly** — every clip ID matches the IDs scored in
  `bench_5k_*_scores.jsonl` and counted in
  `extract_paper_numbers.py`. **All headline numbers in the paper are
  reproducible from these splits.**

For full reconstruction of the original 22,869 raw pool the live
crawl-and-publish pipeline can be re-run from the codebase in this repo;
because the underlying source platforms (especially Bilibili) keep removing
content, the resulting count will drift.

## Schema (`metadata.jsonl`, one JSON object per line)

| Field | Type | Notes |
|-------|------|-------|
| `id` | string (32 hex) | Stable per-clip identifier |
| `source_platform` | enum | `youtube` \| `bilibili` \| `reddit` \| `showcase` |
| `source_url` | URL | Original platform URL (used at re-download time) |
| `source_id` | string | Platform-native ID (e.g. Bilibili `BVxxxx`) |
| `label` | enum | `real` \| `fake` |
| `label_source` | enum | `tier1_dataset` \| `tier1_gallery` \| `tier1_platform_absence` \| `tier2_channel_whitelist` \| `tier2_platform_tag` \| `tier3_llm` |
| `claimed_generator` | string \| null | E.g. `kling21`, `sora2`, `veo3` (null when not declared) |
| `duration_sec` | float \| null | |
| `resolution_w`, `resolution_h` | int \| null | |
| `fps` | float \| null | |
| `file_size_bytes` | int \| null | |
| `blob_sha256` | string \| null | Content hash captured at original crawl time |
| `has_watermark` | bool \| null | |
| `title` | string | Original platform title (no uploader handles) |
| `content_tags` | list[string] | Platform-supplied tags (parsed from JSON) |
| `published_at`, `crawled_at` | ISO-8601 string | Provenance metadata |

## Schema (split files, one JSON object per line)

Each split record is a thin index pointing back at `metadata.jsonl`:

```json
{
  "id": "7c64572361204258b4801fc6547d0ed9",
  "label": "fake",
  "source_platform": "bilibili",
  "source_url": "https://www.bilibili.com/video/BV1h2w1ziEyV",
  "claimed_generator": "kling21"
}
```

`scripts/download_videos.py` reads either the manifest or any split file
and recovers the actual mp4 from `source_url`.

## License

Metadata: **CC-BY-NC 4.0**, following the VidProM precedent. Per-clip
rights remain with the original uploaders.

## Opt-out

24-hour removal guarantee — see `../../docs/OPT_OUT.md`.
