# M0 — frozen monthly slice (submission snapshot)

This directory contains the **M0 manifest** and the train / test / gap-eval
splits used by the paper's headline tables. Metadata only — no video bytes
are redistributed (see top-level `LICENSE` and `../../docs/OPT_OUT.md`).

## Files

| File | Records | Used in |
|------|---------|---------|
| `metadata.jsonl` | 21,504 | M0 composition (paper Sec. 3.2, 4; Tables 2--3) |
| `splits/train.jsonl` | 7,957 | LP / FT training |
| `splits/test.jsonl` | 1,999 | LP / FT evaluation |
| `splits/gap_test.jsonl` | 9,956 | Evaluation pool for existing detectors |

The two LP/FT splits sum to 9,956 — the same population as `gap_test.jsonl`
— and are a stratified-by-(label × generator) 80/20 split of it.

## Composition

| | Clips |
|---|---|
| Total | 21,504 (11,502 generated / 10,002 real) |
| Bilibili | 17,154 (7,152 generated / 10,002 real) |
| Reddit | 3,695 generated |
| YouTube | 574 generated |
| Official galleries (`showcase`) | 81 generated |

Evidence tiers (paper Sec. 4): Tier 1 = 81 (`tier1_gallery`); Tier 2 = 21,359
(`tier2_platform_tag` 7,134, `tier2_channel_whitelist` 4,223, and
`tier1_platform_absence` 10,002 untagged platform reals); Tier 3 = 64
(`tier3_llm`). The `label_source` values keep their collection-time prefixes.

Split files are self-contained: each record carries `id`, `label`, and
`source_url`. 3,179 of the 9,956 `gap_test.jsonl` records, including all
198 Kinetics-400 reals, have no entry in `metadata.jsonl`.

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
