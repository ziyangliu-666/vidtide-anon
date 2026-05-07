# M0 — frozen monthly slice (submission snapshot)

This directory will hold the **M0 manifest**: 22,869 records, metadata only, no video bytes.

## Schema (one JSON object per line)

| Field | Type | Notes |
|-------|------|-------|
| `id` | string (32 hex) | Stable per-clip identifier |
| `source_platform` | enum | `youtube` \| `bilibili` \| `reddit` \| `showcase` |
| `source_url` | URL | Original platform URL (used at re-download time) |
| `source_id` | string | Platform-native ID (e.g. Bilibili `BVxxxx`) |
| `label` | enum | `real` \| `fake` |
| `tier_source` | enum | `tier1_official_showcase` \| `tier2_platform_tag` \| `tier2_channel_whitelist` \| `tier3_keyword_llm` |
| `claimed_generator` | string \| null | E.g. `kling21`, `sora2`, `veo3` (null when not declared) |
| `duration_sec` | float | |
| `resolution_w`, `resolution_h` | int \| null | |
| `fps` | float \| null | |
| `file_size_bytes` | int | |
| `sha256` | string | Content hash captured at original crawl time |
| `title`, `content_tags`, `published_at`, `crawled_at` | strings | Provenance metadata |

## Splits

- `splits/train.jsonl` — 7,957 fine-tune training records.
- `splits/val.jsonl` — held-out validation.
- `splits/test.jsonl` — 5K real / 5K fake gap-evaluation slice (Table~`tab:gap` in the paper).

## License

Metadata: **CC-BY-NC 4.0**, following the VidProM precedent. Per-clip rights remain with the original uploaders.

## Opt-out

24-hour removal guarantee — see `../../docs/OPT_OUT.md`.

---

> **Skeleton notice.** The actual `metadata.jsonl` and `splits/*.jsonl` files will be uploaded to this directory once an additional anonymisation review pass over the title / tag fields is complete (we strip uploader handles and any URL fragments that re-identify named individual creators).
