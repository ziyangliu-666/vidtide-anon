"""Video crawlers — RollingForge tier1/tier2/tier3 ingestion sources.

Adding a new crawler — checklist
================================

Each crawler is a `BaseCrawler` subclass living in its own module under
`server/crawler/<name>.py`. The pipeline picks it up automatically as long as
the conventions below are followed.

1. **Subclass `BaseCrawler`** with a unique `name` and a numeric `tier`.
   Implement `crawl(config) -> Iterator[CrawledVideo]` and
   `estimate_available(config) -> int`.

2. **Decorate with `@register("<name>")`** from `server.crawler.registry`.
   This is the entire wiring step. The runner discovers the class via the
   registry; you do NOT need to edit `server/pipeline/runner.py`.

3. **`CrawledVideo.source_url` MUST be a directly playable URL** when one
   exists (CDN mp4/webm/mov/m4v) — that's what the dashboard's `VideoEmbed`
   feeds into the HTML5 `<video>` element. Storing a marketing-page URL or
   an HTML wrapper page here will break inline playback. If the platform
   only exposes an embed (YouTube, Bilibili), `source_url` can be the
   canonical watch page and you must add a matching `case` in
   `web/src/components/video-embed.tsx` that renders the platform iframe.

4. **Set `claimed_generator`** by importing `extract_generator()` from
   `server.crawler._generator` and feeding it any free text (title, caption,
   description, tags). This is what `ModelWhitelistFilter` checks. Don't
   roll your own model-name regex — extend `_generator.py` instead.

5. **Generate a thumbnail** with
   `from server.crawler._thumbnail import extract_thumbnail` and stash the
   bytes on `CrawledVideo.thumbnail_bytes`. The runner ships them via the
   remote-push payload, the cloud writes them to its volume, and the
   dashboard serves them as static files. Don't try to run ffmpeg server-side
   — the Fly machine is 512 MB and OOM-kills uvicorn on contact with ffmpeg.

6. **Umbrella platforms** (one crawler producing rows from multiple distinct
   vendor sources, e.g. ShowcaseCrawler covers DeepMind Veo + Runway Gen-4
   + Sora + ...) MUST stamp a `<source_platform>:<source_key>` entry into
   `content_tags`. The dashboard's `displayPlatform()` and `/api/stats`
   `by_platform` aggregation auto-expand any tag matching that pattern,
   so the pie chart shows real vendor breakdowns instead of one giant lump.

7. **Add a YAML config block** under `crawl.platforms.<name>` in
   `config/pipeline.yaml`. The runner iterates that map; only entries with
   `enabled: true` are loaded.

8. **Optional deps** (e.g. yt-dlp, playwright, ffmpeg) should be import-time
   try/except guards or `shutil.which(...)` checks so a missing dep disables
   only this crawler, not the whole pipeline. The registry's `load_enabled()`
   will catch `ImportError` and skip cleanly.

That's it. The curated-yawning-wadler refactor moved everything to this
convention so adding a new crawler is registry + config only — no edits
to runner.py, no edits to filter chain, no edits to import_videos.py.
"""
