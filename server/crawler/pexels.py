"""Pexels CC0 real-video crawler.

Pexels hosts royalty-free stock videos under a permissive CC0-like license.
We use Pexels as a second source of real-video negatives (complementing
Kinetics, which is UGC-sourced from YouTube).

API: https://www.pexels.com/api/documentation/
Free tier: 200 req/hour, 20000/month.

Set PEXELS_API_KEY env var before running.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

import requests

from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_POPULAR_URL = "https://api.pexels.com/videos/popular"


@register("pexels")
class PexelsCrawler(BaseCrawler):
    """Sample Pexels CC0 videos for real-class negatives."""

    name = "pexels"
    tier = 1

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        api_key = os.environ.get("PEXELS_API_KEY") or config.get("api_key")
        if not api_key:
            logger.warning(
                "PexelsCrawler: no API key set (env PEXELS_API_KEY) — skipping"
            )
            return

        max_videos = config.get("max_videos", 200)
        search_queries = config.get(
            "search_queries",
            ["nature", "city", "people", "food", "animals", "sports",
             "travel", "technology", "business", "lifestyle"],
        )
        per_page = config.get("per_page", 40)
        min_duration = config.get("min_duration", 3)
        max_duration = config.get("max_duration", 60)

        headers = {"Authorization": api_key}
        yielded = 0
        seen_ids: set[str] = set()

        for query in search_queries:
            if yielded >= max_videos:
                break

            logger.info("PexelsCrawler: searching '%s'", query)
            try:
                resp = requests.get(
                    PEXELS_SEARCH_URL,
                    params={"query": query, "per_page": per_page, "page": 1},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                logger.warning("PexelsCrawler: search failed for '%s'", query, exc_info=True)
                continue

            for video in data.get("videos", []):
                if yielded >= max_videos:
                    break

                vid = str(video.get("id", ""))
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)

                duration = video.get("duration")
                if duration and (duration < min_duration or duration > max_duration):
                    continue

                # Pick the HD 720p-or-lower file
                files = video.get("video_files") or []
                best_url = None
                for f in sorted(files, key=lambda x: x.get("height") or 0):
                    h = f.get("height") or 0
                    if h <= 720 and f.get("file_type", "").startswith("video/"):
                        best_url = f.get("link")
                if not best_url and files:
                    best_url = files[0].get("link")
                if not best_url:
                    continue

                yield CrawledVideo(
                    source_platform="pexels",
                    source_url=best_url,
                    source_id=vid,
                    label="real",
                    label_source="tier1_dataset",
                    title=(video.get("user", {}) or {}).get("name", "Pexels video"),
                    claimed_generator=None,
                    content_tags=[f"pexels:{query}"],
                    raw_metadata={
                        "pexels_query": query,
                        "width": video.get("width"),
                        "height": video.get("height"),
                    },
                    download_url=best_url,
                    thumbnail_url=video.get("image"),
                    duration_sec=float(duration) if duration else None,
                    resolution_w=video.get("width"),
                    resolution_h=video.get("height"),
                    fps=None,
                )
                yielded += 1

            time.sleep(1.0)  # rate limit courtesy

        logger.info("PexelsCrawler: yielded %d videos", yielded)

    def estimate_available(self, config: dict) -> int:
        return config.get("max_videos", 200)
