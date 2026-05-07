# Crawlers

Per-platform harvesters implementing the `BaseCrawler` interface:

```python
class BaseCrawler:
    name: str
    tier: int  # 1 = official showcase, 2 = platform tag, 3 = keyword + LLM

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        ...
```

| Module | Platform | Method | Ground-truth source |
|--------|----------|--------|---------------------|
| `youtube.py` | YouTube | `yt-dlp` search + AI-disclosure tag | Mandatory AI disclosure (since 2024) |
| `bilibili.py` | Bilibili | `yt-dlp` `bilisearch:` + Chinese AI keywords + `argue_info` tag | China AI labelling regulation (Sept 2025) |
| `reddit.py` | Reddit | Public JSON API (r/aivideo, r/sora, r/StableDiffusion, ...) | Subreddit context + user self-report |
| `showcase.py` | Official galleries | Per-vendor scrapers (Pika / Kling / Runway / ...) | T1 — definitionally AI-generated |

> **Skeleton notice.** Stub `.py` files documenting each interface land in the next batch; full crawler implementations follow once the per-platform Terms-of-Service compliance notes (rate limits, attribution, opt-out hooks) are finalised in `docs/PIPELINE.md`.
