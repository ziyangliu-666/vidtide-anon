# Filters

Sequential filter stack applied to candidates emitted by the crawlers. All
implementations live in [`server/filter/`](../server/filter/) and are
re-exported here.

| Filter | Module | Purpose |
|--------|--------|---------|
| Quality | `server/filter/quality_filter.py` | Drops clips below resolution / duration / fps thresholds (default: ≥720p, 3–300 s, ffprobe structural validation) |
| Tag-tier | `server/filter/tag_filter.py` | Routes videos by configured label tier (T1 / T2 / T3) |
| LLM verification | `server/filter/llm_filter.py` | GPT-4o title/description verification — separates *actual* AI videos from *tutorials about* AI |
| Generator whitelist | `server/filter/model_whitelist_filter.py` | Drops clips whose `claimed_generator` is outside the paper's named-generator pool |

Each implements `BaseFilter` (`server/filter/base.py`) — a single
`apply(video, config) -> FilterDecision` method. Per-filter config keys
are documented inline at the top of each module.

The order, thresholds, and on/off switches are wired up in
`config/pipeline.yaml.example`.
