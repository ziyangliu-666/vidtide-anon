# Crawlers

Per-platform harvesters that emit `CrawledVideo` records into the pipeline.
The full implementations live in [`server/crawler/`](../server/crawler/);
this top-level package re-exports them so the directory layout matches the
README's tree (`from crawlers import BilibiliCrawler` works).

| Module | Platform | Method | Ground-truth tier |
|--------|----------|--------|-------------------|
| `server/crawler/youtube.py` | YouTube | `yt-dlp` search + AI-disclosure tag | T2 — mandatory AI disclosure (since 2024) |
| `server/crawler/bilibili.py` | Bilibili (fakes) | `yt-dlp` `bilisearch:` + Chinese AI keywords + `argue_info` AI tag | T2 — China AI labelling regulation (Sept 2025) |
| `server/crawler/bilibili_real.py` | Bilibili (reals) | Channel whitelist + `argue_info` absence | T1 — platform-absence-of-AI-tag |
| `server/crawler/reddit.py` | Reddit | Public JSON API (r/aivideo, r/sora, r/StableDiffusion, …) | T2/T3 — subreddit context + LLM verification |
| `server/crawler/showcase.py` | Official galleries (Pika, Kling, Runway, Dreamina, Veo, Sora, …) | Per-vendor scrapers | T1 — definitionally AI-generated |
| `server/crawler/douyin.py`, `kuaishou.py` | (experimental) Douyin / Kuaishou | yt-dlp + tag detection | T2 |
| `server/crawler/kinetics.py`, `pexels.py` | Real-video supplementation pools | Direct dataset mirror | T1 — pre-AI-era curated reals |

All crawlers implement the `BaseCrawler` interface defined in
[`server/crawler/base.py`](../server/crawler/base.py); a new platform
requires only a YAML configuration entry (see `config/pipeline.yaml.example`)
and a subclass.

Per-platform Terms-of-Service compliance, rate-limit defaults, and opt-out
hooks are documented in [`../docs/PIPELINE.md`](../docs/PIPELINE.md).
