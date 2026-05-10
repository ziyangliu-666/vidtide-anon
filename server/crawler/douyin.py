"""Douyin (抖音) crawler — AI videos via search, platform AI-label gated.

Douyin is ByteDance's Chinese short-video platform (the domestic version of
TikTok). Under China's《互联网信息服务深度合成管理规定》, AI-generated content
must be labeled. The crawler searches for generic keywords and filters
results by the platform-side AI label — same strategy as the bilibili
argue_msg gate.

**Anti-bot**: Douyin's web API uses a-bogus / X-Bogus signature on every
request. The signing algorithm is reverse-engineered in several open-source
libraries (douyin-a-bogus, etc.). We use a minimal implementation here.

**Discovery**: keyword search → paginated results → AI label filter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Iterator

import requests

from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator name extraction (reused from bilibili pattern)
# ---------------------------------------------------------------------------

_GENERATOR_PATTERNS = [
    (r"(?i)sora\s*2", "sora2"),
    (r"(?i)sora", "sora2"),
    (r"(?i)可灵|kling", "kling21"),
    (r"(?i)即梦|dreamina", "dreamina3"),
    (r"(?i)runway", "runway-gen4"),
    (r"(?i)veo\s*3", "veo3"),
    (r"(?i)hailuo|海螺|minimax", "hailuo"),
    (r"(?i)pika", "pika2"),
    (r"(?i)luma|dream\s*machine", "luma"),
    (r"(?i)hunyuan|混元", "hunyuan"),
    (r"(?i)通义万相|wanx|wan\s*2", "wan21"),
    (r"(?i)stable\s*video|svd", "stable_video_diffusion"),
    (r"(?i)seedance", "seedance"),
    (r"(?i)vidu", "vidu"),
    (r"(?i)pixverse", "pixverse"),
    (r"(?i)gen[\s-]?mo", "genmo"),
]


def _extract_generator(text: str) -> str | None:
    for pattern, name in _GENERATOR_PATTERNS:
        if re.search(pattern, text):
            return name
    return None


# ---------------------------------------------------------------------------
# Douyin session + signing
# ---------------------------------------------------------------------------

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_MIN_CALL_INTERVAL = float(os.environ.get("DOUYIN_MIN_INTERVAL", "2.0"))


class _DouyinSession:
    """Manages cookies, tokens, and rate limiting for Douyin API calls."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _UA,
            "Referer": "https://www.douyin.com/",
        })
        self._last_call = 0.0
        self._ms_token: str | None = None

    def bootstrap(self) -> None:
        """Visit douyin.com to acquire initial cookies (ttwid, msToken)."""
        try:
            resp = self._session.get(
                "https://www.douyin.com/",
                timeout=15,
                allow_redirects=True,
            )
            resp.raise_for_status()
            # Extract msToken from cookies
            self._ms_token = self._session.cookies.get("msToken")
            logger.info(
                "DouyinSession: bootstrapped, msToken=%s",
                "present" if self._ms_token else "missing",
            )
        except Exception:
            logger.warning("DouyinSession: bootstrap failed", exc_info=True)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        self._last_call = time.time()

    def search(
        self,
        keyword: str,
        offset: int = 0,
        count: int = 20,
        sort_type: int = 0,  # 0=综合, 1=最新, 2=最热
    ) -> dict:
        """Search Douyin for videos matching keyword.

        Returns the raw JSON response dict. The caller is responsible for
        extracting video items and checking AI labels.
        """
        self._rate_limit()

        params = {
            "keyword": keyword,
            "search_channel": "aweme_video_web",
            "sort_type": sort_type,
            "publish_time": 0,  # 0=不限, 1=一天内, 7=一周内
            "offset": offset,
            "count": count,
            "cookie_enabled": "true",
            "platform": "PC",
            "aid": "6383",
            "channel": "channel_pc_web",
            "version_code": "170400",
            "version_name": "17.4.0",
        }

        if self._ms_token:
            params["msToken"] = self._ms_token

        # The general search endpoint does NOT require a_bogus signing
        # (confirmed in NanmiCoder/MediaCrawler: explicit bypass for
        # /v1/web/general/search). This is simpler than bilibili's WBI.

        try:
            resp = self._session.get(
                "https://www.douyin.com/aweme/v1/web/general/search/single/",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning(
                "DouyinSession: search failed for '%s' offset=%d",
                keyword, offset, exc_info=True,
            )
            return {}


# ---------------------------------------------------------------------------
# AI label detection
# ---------------------------------------------------------------------------

def _has_ai_label(aweme: dict) -> bool:
    """Check if a Douyin video (aweme) has the platform AI-generated label.

    The label location varies by API version. Known paths:
    - aweme.label_top_text containing "AI生成"
    - aweme.video_tag containing AI-related tags
    - aweme.aigc_info or aweme.ai_label fields
    - aweme.status.review_result containing AIGC flags

    We check multiple paths for robustness.
    """
    # Path 1: label_top_text (most common)
    label_top = aweme.get("label_top_text") or {}
    if isinstance(label_top, dict):
        label_text = label_top.get("text", "")
    elif isinstance(label_top, str):
        label_text = label_top
    else:
        label_text = ""
    if "AI" in label_text or "深度合成" in label_text or "人工智能" in label_text:
        return True

    # Path 2: aigc_info field
    aigc = aweme.get("aigc_info") or {}
    if aigc.get("aigc_label_type") or aigc.get("is_aigc"):
        return True

    # Path 3: video_tag list
    for tag in (aweme.get("video_tag") or []):
        tag_name = tag.get("tag_name", "") if isinstance(tag, dict) else str(tag)
        if "AI" in tag_name or "AIGC" in tag_name or "深度合成" in tag_name:
            return True

    # Path 4: caption / desc text containing the regulatory notice
    desc = aweme.get("desc", "") or ""
    if "该内容由AI生成" in desc or "深度合成内容" in desc:
        return True

    # Path 5: search result level label
    for cell_label in (aweme.get("cell_label_list") or []):
        if isinstance(cell_label, dict):
            lt = cell_label.get("label_text", "")
            if "AI" in lt or "深度合成" in lt:
                return True

    return False


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

@register("douyin")
class DouyinCrawler(BaseCrawler):
    """Douyin video crawler using web search + AI label gate."""

    name = "douyin"
    tier = 2

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        max_videos = config.get("max_videos", 100)
        search_queries = config.get("search_queries", [])
        pages_per_query = config.get("pages_per_query", 5)
        page_size = config.get("page_size", 20)
        min_duration = config.get("min_duration", 3)
        max_duration = config.get("max_duration", 600)

        session = _DouyinSession()
        session.bootstrap()

        seen_ids: set[str] = set()
        yielded = 0

        for query in search_queries:
            if yielded >= max_videos:
                break

            for page in range(pages_per_query):
                if yielded >= max_videos:
                    break

                offset = page * page_size
                logger.info(
                    "DouyinCrawler: search '%s' offset=%d",
                    query, offset,
                )

                data = session.search(
                    keyword=query,
                    offset=offset,
                    count=page_size,
                    sort_type=1,  # 最新
                )

                items = data.get("data") or []
                if not items:
                    logger.debug(
                        "DouyinCrawler: empty results for '%s' offset=%d",
                        query, offset,
                    )
                    break

                for item in items:
                    aweme = item.get("aweme_info") or item
                    aweme_id = str(aweme.get("aweme_id", ""))
                    if not aweme_id or aweme_id in seen_ids:
                        continue
                    seen_ids.add(aweme_id)

                    # AI label gate
                    if not _has_ai_label(aweme):
                        continue

                    # Duration filter
                    duration = None
                    video_info = aweme.get("video") or {}
                    dur_raw = video_info.get("duration")
                    if dur_raw is not None:
                        try:
                            # Douyin duration is in milliseconds
                            duration = float(dur_raw) / 1000.0
                        except (ValueError, TypeError):
                            pass

                    if duration is not None:
                        if duration < min_duration or duration > max_duration:
                            continue

                    # Extract metadata
                    desc = aweme.get("desc", "") or ""
                    author = aweme.get("author", {}) or {}
                    author_name = author.get("nickname", "")

                    # Resolution
                    width = video_info.get("width")
                    height = video_info.get("height")

                    # Thumbnail
                    cover = video_info.get("cover") or {}
                    thumb_urls = cover.get("url_list") or []
                    thumbnail_url = thumb_urls[0] if thumb_urls else None

                    # Source URL (playable)
                    play_addr = video_info.get("play_addr") or {}
                    play_urls = play_addr.get("url_list") or []
                    source_url = play_urls[0] if play_urls else f"https://www.douyin.com/video/{aweme_id}"

                    # Published time
                    create_time = aweme.get("create_time")
                    published_at = None
                    if create_time:
                        try:
                            published_at = datetime.fromtimestamp(
                                int(create_time), tz=timezone.utc
                            ).isoformat()
                        except (ValueError, TypeError, OSError):
                            pass

                    claimed_generator = _extract_generator(desc)

                    yield CrawledVideo(
                        source_platform="douyin",
                        source_url=source_url,
                        source_id=aweme_id,
                        label="fake",
                        label_source="tier2_platform_tag",
                        title=desc[:200] if desc else None,
                        claimed_generator=claimed_generator,
                        content_tags=[f"author:{author_name}"] if author_name else [],
                        published_at=published_at,
                        raw_metadata=aweme,
                        download_url=source_url,
                        thumbnail_url=thumbnail_url,
                        duration_sec=duration,
                        resolution_w=int(width) if width else None,
                        resolution_h=int(height) if height else None,
                        fps=None,
                    )
                    yielded += 1

                    if yielded % 50 == 0:
                        logger.info(
                            "DouyinCrawler: %d videos yielded so far",
                            yielded,
                        )

        logger.info("DouyinCrawler: yielded %d videos total", yielded)

    def estimate_available(self, config: dict) -> int:
        queries = config.get("search_queries", [])
        pages = config.get("pages_per_query", 5)
        page_size = config.get("page_size", 20)
        max_videos = config.get("max_videos", 100)
        return min(len(queries) * pages * page_size, max_videos)
