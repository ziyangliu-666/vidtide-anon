"""Kuaishou (快手) crawler — AI videos via search, platform AI-label gated.

Kuaishou is a major Chinese short-video platform. Under the same Chinese
AI regulation as bilibili/Douyin, AI-generated content must be labeled.
The crawler searches for generic keywords and filters by the platform-side
AI label.

**Anti-bot**: Kuaishou uses cookie-based auth + request signing.
**Discovery**: keyword search → paginated results → AI label filter.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Iterator

import requests

from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator name extraction
# ---------------------------------------------------------------------------

_GENERATOR_PATTERNS = [
    (r"(?i)sora\s*2", "sora2"),
    (r"(?i)可灵|kling", "kling21"),
    (r"(?i)即梦|dreamina", "dreamina3"),
    (r"(?i)runway", "runway-gen4"),
    (r"(?i)veo\s*3", "veo3"),
    (r"(?i)hailuo|海螺|minimax", "hailuo"),
    (r"(?i)pika", "pika2"),
    (r"(?i)luma|dream\s*machine", "luma"),
    (r"(?i)hunyuan|混元", "hunyuan"),
    (r"(?i)通义万相|wanx|wan\s*2", "wan21"),
    (r"(?i)seedance", "seedance"),
    (r"(?i)vidu", "vidu"),
    (r"(?i)pixverse", "pixverse"),
]


def _extract_generator(text: str) -> str | None:
    for pattern, name in _GENERATOR_PATTERNS:
        if re.search(pattern, text):
            return name
    return None


# ---------------------------------------------------------------------------
# Kuaishou session
# ---------------------------------------------------------------------------

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_MIN_CALL_INTERVAL = float(os.environ.get("KUAISHOU_MIN_INTERVAL", "2.0"))


class _KuaishouSession:
    """Manages cookies and rate limiting for Kuaishou API calls."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _UA,
            "Referer": "https://www.kuaishou.com/",
            "Origin": "https://www.kuaishou.com",
        })
        self._last_call = 0.0

    def bootstrap(self) -> None:
        """Visit kuaishou.com to acquire cookies."""
        try:
            resp = self._session.get(
                "https://www.kuaishou.com/",
                timeout=15,
                allow_redirects=True,
            )
            resp.raise_for_status()
            logger.info(
                "KuaishouSession: bootstrapped, cookies=%d",
                len(self._session.cookies),
            )
        except Exception:
            logger.warning("KuaishouSession: bootstrap failed", exc_info=True)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        self._last_call = time.time()

    def search(self, keyword: str, pcursor: str = "", count: int = 20) -> dict:
        """Search Kuaishou for videos.

        Kuaishou's web search uses a GraphQL-style POST endpoint.
        Returns the raw JSON response.
        """
        self._rate_limit()

        # Kuaishou web search uses a POST to /graphql with a query
        payload = {
            "operationName": "visionSearchPhoto",
            "variables": {
                "keyword": keyword,
                "pcursor": pcursor,
                "page": "search",
            },
            "query": (
                "query visionSearchPhoto($keyword: String, $pcursor: String, "
                "$page: String) {\n  visionSearchPhoto(keyword: $keyword, "
                "pcursor: $pcursor, page: $page) {\n    result\n    "
                "llsid\n    webPageArea\n    feeds {\n      type\n      "
                "author {\n        id\n        name\n        following\n"
                "        headerUrl\n      }\n      tags\n      photo {\n"
                "        id\n        duration\n        caption\n        "
                "photoUrl\n        coverUrl\n        photoH265Url\n        "
                "manifest {\n          mediaType\n          businessType\n"
                "          version\n          adaptationSet {\n            "
                "id\n            duration\n            representation {\n"
                "              id\n              defaultSelect\n              "
                "backupUrl\n              shortUrl\n              width\n"
                "              height\n              qualityLabel\n            "
                "}\n          }\n        }\n        videoResource\n        "
                "width\n        height\n        realLikeCount\n        "
                "viewCount\n        timestamp\n        animatedCoverUrl\n"
                "        stereoType\n        videoRatio\n        "
                "aigcInfo {\n          aigcLabelType\n          "
                "aigcTagList\n        }\n      }\n    }\n    pcursor\n"
                "  }\n}\n"
            ),
        }

        try:
            resp = self._session.post(
                "https://www.kuaishou.com/graphql",
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning(
                "KuaishouSession: search failed for '%s'",
                keyword, exc_info=True,
            )
            return {}


# ---------------------------------------------------------------------------
# AI label detection
# ---------------------------------------------------------------------------

def _has_ai_label(photo: dict) -> bool:
    """Check if a Kuaishou photo/video has the AI-generated label.

    Known paths:
    - photo.aigcInfo.aigcLabelType (non-zero = AI)
    - photo.aigcInfo.aigcTagList (non-empty = AI)
    - photo.caption containing regulatory text
    """
    aigc = photo.get("aigcInfo") or {}

    # aigcLabelType: 0 = not AI, >0 = AI generated
    label_type = aigc.get("aigcLabelType")
    if label_type and int(label_type) > 0:
        return True

    # aigcTagList
    tags = aigc.get("aigcTagList") or []
    if tags:
        return True

    # Fallback: caption text
    caption = photo.get("caption") or ""
    if "AI生成" in caption or "深度合成" in caption:
        return True

    return False


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

@register("kuaishou")
class KuaishouCrawler(BaseCrawler):
    """Kuaishou video crawler using web search + AI label gate."""

    name = "kuaishou"
    tier = 2

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        max_videos = config.get("max_videos", 100)
        search_queries = config.get("search_queries", [])
        pages_per_query = config.get("pages_per_query", 5)
        min_duration = config.get("min_duration", 3)
        max_duration = config.get("max_duration", 600)

        session = _KuaishouSession()
        session.bootstrap()

        seen_ids: set[str] = set()
        yielded = 0

        for query in search_queries:
            if yielded >= max_videos:
                break

            pcursor = ""
            for page in range(pages_per_query):
                if yielded >= max_videos:
                    break

                logger.info(
                    "KuaishouCrawler: search '%s' page=%d",
                    query, page + 1,
                )

                data = session.search(keyword=query, pcursor=pcursor)

                vision = data.get("data", {}).get("visionSearchPhoto") or {}
                feeds = vision.get("feeds") or []
                pcursor = vision.get("pcursor") or ""

                if not feeds:
                    break

                for feed in feeds:
                    photo = feed.get("photo") or {}
                    photo_id = str(photo.get("id", ""))
                    if not photo_id or photo_id in seen_ids:
                        continue
                    seen_ids.add(photo_id)

                    # AI label gate
                    if not _has_ai_label(photo):
                        continue

                    # Duration filter
                    duration = None
                    dur_raw = photo.get("duration")
                    if dur_raw is not None:
                        try:
                            # Kuaishou duration is in milliseconds
                            duration = float(dur_raw) / 1000.0
                        except (ValueError, TypeError):
                            pass

                    if duration is not None:
                        if duration < min_duration or duration > max_duration:
                            continue

                    # Extract metadata
                    caption = photo.get("caption") or ""
                    author = feed.get("author") or {}
                    author_name = author.get("name", "")

                    # Resolution
                    width = photo.get("width")
                    height = photo.get("height")

                    # Thumbnail
                    thumbnail_url = photo.get("coverUrl")

                    # Source URL
                    source_url = photo.get("photoUrl") or f"https://www.kuaishou.com/short-video/{photo_id}"

                    # Published time
                    timestamp = photo.get("timestamp")
                    published_at = None
                    if timestamp:
                        try:
                            published_at = datetime.fromtimestamp(
                                int(timestamp) / 1000, tz=timezone.utc
                            ).isoformat()
                        except (ValueError, TypeError, OSError):
                            pass

                    claimed_generator = _extract_generator(caption)

                    yield CrawledVideo(
                        source_platform="kuaishou",
                        source_url=source_url,
                        source_id=photo_id,
                        label="fake",
                        label_source="tier2_platform_tag",
                        title=caption[:200] if caption else None,
                        claimed_generator=claimed_generator,
                        content_tags=[f"author:{author_name}"] if author_name else [],
                        published_at=published_at,
                        raw_metadata=photo,
                        download_url=source_url,
                        thumbnail_url=thumbnail_url,
                        duration_sec=duration,
                        resolution_w=int(width) if width else None,
                        resolution_h=int(height) if height else None,
                        fps=None,
                    )
                    yielded += 1

                if not pcursor:
                    break

        logger.info("KuaishouCrawler: yielded %d videos total", yielded)

    def estimate_available(self, config: dict) -> int:
        queries = config.get("search_queries", [])
        pages = config.get("pages_per_query", 5)
        max_videos = config.get("max_videos", 100)
        return min(len(queries) * pages * 20, max_videos)
