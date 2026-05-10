"""ShowcaseCrawler — tier1 ingestion from official model marketing pages.

Each configured "source" is one vendor showcase page (Sora 2, Veo 3, Runway
Gen-4, etc.). The crawler GETs the page with `requests`, regex-extracts CDN
video URLs from the HTML, caches the raw HTML for diff/debug, and yields one
`CrawledVideo` per unique URL with a hardcoded `claimed_generator` so the
provenance tier is undisputed.

Design notes:
- **No new dependencies**: pure `requests` + stdlib `re` + `urllib.robotparser`.
  No BeautifulSoup, no Playwright. The trade-off is that purely JS-rendered
  galleries (Kling community, Luma community, etc.) need a v2 with Playwright.
- **One-shot seed semantics**: marketing pages have ~5-15 demos each. Re-runs
  hit `source_id` dedup in the runner. This is fine — showcase content is the
  tier1 gold-standard provenance for the bench, not a sustained crawl.
- **Per-source parsers**: each source declares a `parser` key in config. Today
  the universal `generic_mp4` parser is enough for all four bootstrap sources;
  vendor-specific overrides slot in later if a page redesign breaks extraction.
- **HTML caching**: every fetch dumps the response body to
  `data/cache/showcase/<source_key>/<YYYY-MM-DD>.html` so a future failure can
  be diffed against the last known-good snapshot.
- **robots.txt**: respected by default per host, with stdlib `RobotFileParser`.
  Set `respect_robots: false` in config to override (use only with explicit
  vendor permission).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from server.crawler._thumbnail import extract_thumbnail
from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)

_USER_AGENT = "RollingForge/0.1 (research crawler; benchmark curation)"
_HEADERS = {"User-Agent": _USER_AGENT}
# Marketing pages on heavy CDNs (Runway, Vercel-fronted vendor sites) routinely
# need >30s on first hit. 60s default + one retry handles the long tail.
_REQUEST_TIMEOUT = 60
_RETRY_DELAY_SEC = 2


# ----------------------------------------------------------------------
# Per-source parsers
# ----------------------------------------------------------------------

# Catches direct video CDN URLs in HTML, JSON-embedded blobs, and srcset
# entries. Two-part pattern:
#
#   path:   [^\s"'<>()\\?#&]+   — no whitespace, quotes, brackets, backslash,
#                                 and (critically) no `?#&` so the path can't
#                                 swallow embedded query strings. This stops
#                                 the regex from matching things like
#                                 `comparison/index.html?left=...video.mp4`
#                                 where `.mp4` only appears inside a query
#                                 parameter — DeepMind's Veo page has these
#                                 viewer URLs and we'd otherwise pick them
#                                 up as if they were direct video files.
#
#   query:  ?[^\s"'<>()\\#]*    — optional, allows `&` so signed-URL tokens
#                                 with multiple params still work.
_VIDEO_URL_RE = re.compile(
    r"""https?://[^\s"'<>()\\?#&]+\.(?:mp4|webm|m4v|mov)(?:\?[^\s"'<>()\\#]*)?""",
    re.IGNORECASE,
)


def _parse_generic_video(html: str, src: dict) -> list[dict]:
    """Universal parser: extract every direct .mp4/.webm/.m4v/.mov URL.

    Works for static marketing pages where vendors embed `<video src=...>`
    or JSON-encoded video URLs (Sora 2 index, DeepMind Veo, Runway research).
    Returns one dict per unique URL with `cdn_url` populated; title and
    thumbnail are unknown for the generic parser.
    """
    # JSON-escaped slashes (`\/`) appear in inline `<script>` blobs on most
    # modern marketing pages. Unescape them before regex so the URL match
    # actually fires on script-embedded videos.
    normalized = html.replace("\\/", "/")
    seen: set[str] = set()
    clips: list[dict] = []
    for match in _VIDEO_URL_RE.finditer(normalized):
        url = match.group(0)
        if url in seen:
            continue
        seen.add(url)
        clips.append(
            {
                "cdn_url": url,
                "title": None,
                "thumbnail_url": None,
            }
        )
    return clips


_PARSERS: dict[str, Callable[[str, dict], list[dict]]] = {
    "generic_video": _parse_generic_video,
}


# ----------------------------------------------------------------------
# robots.txt cache
# ----------------------------------------------------------------------


_ROBOTS_CACHE: dict[str, RobotFileParser] = {}


def _robots_allow(url: str) -> bool:
    """Check whether `_USER_AGENT` is allowed to fetch *url* per robots.txt.

    On any error (DNS failure, robots.txt 404, malformed file) we fall back
    to "allowed" — this matches `urllib.robotparser` defaults and avoids
    a single broken robots.txt killing the whole crawl.
    """
    parts = urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    rp = _ROBOTS_CACHE.get(host)
    if rp is None:
        rp = RobotFileParser()
        rp.set_url(f"{host}/robots.txt")
        try:
            rp.read()
        except Exception:
            logger.debug("ShowcaseCrawler: failed to read robots.txt for %s, allowing", host)
            _ROBOTS_CACHE[host] = rp
            return True
        _ROBOTS_CACHE[host] = rp
    try:
        return rp.can_fetch(_USER_AGENT, url)
    except Exception:
        return True


# ----------------------------------------------------------------------
# Crawler
# ----------------------------------------------------------------------


@register("showcase")
class ShowcaseCrawler(BaseCrawler):
    """Crawl official AI-video model showcase pages (tier1 ground-truth)."""

    name = "showcase"
    tier = 1

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        sources: list[dict] = config.get("sources", [])
        max_videos: int = config.get("max_videos", 80)
        cache_root = Path(config.get("cache_dir", "data/cache/showcase"))
        respect_robots: bool = config.get("respect_robots", True)

        if not sources:
            logger.warning("ShowcaseCrawler: no sources configured, nothing to crawl")
            return

        yielded = 0
        for src in sources:
            if yielded >= max_videos:
                break

            key = src.get("key", "?")
            try:
                clips = self._fetch_source(src, cache_root, respect_robots)
            except Exception:
                logger.warning(
                    "ShowcaseCrawler: failed to fetch source=%s",
                    key,
                    exc_info=True,
                )
                continue

            logger.info(
                "ShowcaseCrawler: fetched source=%s, parsed %d clip(s)",
                key,
                len(clips),
            )

            for clip in clips:
                if yielded >= max_videos:
                    break
                yield self._to_crawled_video(clip, src)
                yielded += 1

        logger.info("ShowcaseCrawler: yielded %d videos total", yielded)

    def estimate_available(self, config: dict) -> int:
        return config.get("max_videos", 80)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_source(src: dict, cache_root: Path, respect_robots: bool) -> list[dict]:
        url = src.get("url")
        if not url:
            logger.warning("ShowcaseCrawler: source %s missing 'url'", src.get("key"))
            return []

        # Per-source override: a source can opt out of robots.txt by setting
        # `respect_robots: false`, which only takes effect when paired with
        # explicit user intent (the global flag also has to be true OR the
        # source-level flag has to be false).
        source_robots = src.get("respect_robots", respect_robots)
        if source_robots and not _robots_allow(url):
            logger.warning(
                "ShowcaseCrawler: robots.txt disallows %s for %s, skipping",
                _USER_AGENT,
                url,
            )
            return []

        timeout = src.get("timeout_sec", _REQUEST_TIMEOUT)
        html: str | None = None
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=timeout)
                resp.raise_for_status()
                html = resp.text
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt == 1:
                    logger.info(
                        "ShowcaseCrawler: %s on %s, retrying once",
                        type(exc).__name__,
                        src.get("key"),
                    )
                    time.sleep(_RETRY_DELAY_SEC)
                    continue
                raise
        if html is None:  # defensive — loop should either set html or re-raise
            raise last_exc or RuntimeError("ShowcaseCrawler: unreachable fetch state")

        # Cache raw HTML for diff/debug. Best-effort: don't fail the crawl if
        # the cache write fails (e.g. read-only fs).
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            cache_path = cache_root / src.get("key", "unknown") / f"{today}.html"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        except OSError:
            logger.debug("ShowcaseCrawler: failed to cache HTML for %s", src.get("key"))

        parser_name = src.get("parser", "generic_video")
        parser = _PARSERS.get(parser_name)
        if parser is None:
            logger.warning(
                "ShowcaseCrawler: unknown parser '%s' for source=%s",
                parser_name,
                src.get("key"),
            )
            return []
        return parser(html, src)

    @staticmethod
    def _to_crawled_video(clip: dict, src: dict) -> CrawledVideo:
        cdn_url = clip["cdn_url"]
        # SHA1-based source_id gives stable dedup across re-runs even if the
        # marketing page reorders or paginates clips.
        sid = hashlib.sha1(cdn_url.encode("utf-8")).hexdigest()[:16]
        # source_url MUST be the playable CDN URL — that's what the dashboard's
        # VideoEmbed renders into the <video> element. The marketing page URL
        # (provenance link) is preserved as a content_tag so we can still cite
        # it during human review.
        showcase_page = src.get("url", "")
        # Generate the poster frame inline with the crawl via the shared
        # helper (server.crawler._thumbnail.extract_thumbnail). Local ffmpeg,
        # ~3-7s per clip on a typical desktop; degrades to None on any
        # failure (the dashboard then shows a placeholder).
        thumbnail_bytes = extract_thumbnail(cdn_url)
        return CrawledVideo(
            source_platform="showcase",
            source_url=cdn_url,
            source_id=sid,
            label="fake",
            label_source="tier1_gallery",
            title=clip.get("title") or src.get("key"),
            claimed_generator=src.get("model"),
            content_tags=[
                f"showcase:{src.get('key', 'unknown')}",
                f"model:{src.get('model', 'unknown')}",
                f"page:{showcase_page}",
            ],
            published_at=None,
            raw_metadata={
                "cdn_url": cdn_url,
                "source_key": src.get("key"),
                "showcase_url": showcase_page,
            },
            video_bytes=None,
            download_url=cdn_url,
            thumbnail_url=clip.get("thumbnail_url"),
            thumbnail_bytes=thumbnail_bytes,
            duration_sec=None,
            resolution_w=None,
            resolution_h=None,
            fps=None,
        )
