"""Bilibili same-platform real-video crawler (historical mode).

Simpler approach than per-video argue_msg inversion: scope the search to
videos published BEFORE Bilibili's AI-disclosure feature existed. No live
AI-tag check needed — temporal priority guarantees absence of platform
AI labels. Cuts per-candidate HTTP cost roughly in half.

Cutoff: 2023-01-01. Bilibili's AI-disclosure tagging didn't ship until
late 2023; anything older is safely pre-AI-era. A small residual noise
floor (hand-edited old uploads, re-uploads) is acceptable for a real-side
negative class at this scale.

"URL-only" here means only the mp4 blob is deferred. Thumbnails
(~10 KB each) are fetched inline during crawl so the review UI shows
previews — the Bilibili CDN hotlink-blocks cross-origin fetches, so
the browser can't pull them later.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

from server.crawler.base import BaseCrawler, CrawledVideo
from server.crawler.registry import register
from server.crawler import bilibili as bili_module

logger = logging.getLogger(__name__)

# Pre-AI-disclosure cutoff. Sora / Gen-2 / Pika / etc. didn't have
# platform-side disclosure tags before this. Unix seconds.
_DEFAULT_PUBTIME_END = int(
    datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
)


@register("bilibili_real")
class BilibiliRealCrawler(BaseCrawler):
    """Pre-2023 bilibili videos (guaranteed non-AI by date) — real negatives."""

    name = "bilibili_real"
    tier = 1

    def crawl(self, config: dict) -> Iterator[CrawledVideo]:
        pcfg = dict(config)
        pcfg["historical_mode"] = True
        pcfg["skip_thumbnails"] = config.get("skip_thumbnails", False)
        pcfg.setdefault("pubtime_end_sec", _DEFAULT_PUBTIME_END)

        # Content categories mirroring the AI-side distribution so the
        # real/fake sets stay topic-balanced rather than topic-confounded.
        pcfg.setdefault("search_queries", [
            "短片", "微电影", "动画短片", "剧情短片",
            "搞笑短片", "悬疑短片", "温馨短片",
            "MV", "音乐视频", "翻唱", "舞蹈",
            "古风", "汉服", "武侠", "历史",
            "三国演义", "西游记", "红楼梦",
            "特效", "创意视频",
            "鬼畜", "整活", "名场面",
            "老照片", "修复", "变装",
            "婚礼", "毕业", "童年回忆",
        ])
        pcfg.setdefault("max_videos", 1000)
        pcfg.setdefault("pages_per_query", 5)
        pcfg.setdefault("page_size", 50)
        pcfg.setdefault("min_duration", 3)
        pcfg.setdefault("max_duration", 120)
        # Search by scores (most-liked all-time) since we're looking at
        # historical content — pubdate-desc would surface the noisiest
        # uploads right at the cutoff.
        pcfg.setdefault("search_order_modes", ["scores", "click"])

        yield from bili_module.BilibiliCrawler().crawl(pcfg)

    def estimate_available(self, config: dict) -> int:
        return config.get("max_videos", 1000)
