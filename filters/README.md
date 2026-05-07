# Filters

Sequential filter stack applied to candidates emitted by the crawlers.

| Filter | Purpose | Configurable |
|--------|---------|--------------|
| `quality_filter.py` | Drops clips below resolution / duration / fps thresholds | min_resolution, min/max_duration, min_fps |
| `tag_filter.py` | Keeps only the configured label tiers (T1 / T2 / T3) | tiers_accepted |
| `llm_filter.py` | GPT-4o title/description verification — separates *actual* AI videos from *tutorials about* AI | model, confidence_threshold, on/off |

Each implements:

```python
class BaseFilter:
    name: str
    def apply(self, video: CrawledVideo, config: dict) -> FilterDecision:
        ...
```

> **Skeleton notice.** Implementations follow in the next upload batch (see top-level `README.md`).
