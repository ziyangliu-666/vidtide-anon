#!/usr/bin/env python3
"""Backfill videos.published_at for rows crawled before migration 0006.

Before the published_at column existed, the runner stored crawled_at
(when WE scraped) but threw away the platform-reported upload date.
This script fetches the real publication timestamp from each platform's
detail API and writes it to videos.published_at.

Per-video commit + skip-if-already-set → crash-safe + resumable.

Usage:
  python scripts/backfill_published_at.py
  python scripts/backfill_published_at.py --platform bilibili --limit 100
  python scripts/backfill_published_at.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.db.database import SessionLocal  # noqa: E402
from server.db.migrations import run_migrations  # noqa: E402
from server.db.models import Video  # noqa: E402
from server.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

_BILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}


def backfill_bilibili(video: Video) -> datetime | None:
    """Fetch publication timestamp from Bilibili view API."""
    try:
        resp = requests.get(
            "https://api.bilibili.com/x/web-interface/view",
            params={"bvid": video.source_id},
            headers=_BILI_HEADERS,
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", {})
        pubdate = data.get("pubdate")
        if pubdate:
            return datetime.fromtimestamp(float(pubdate), tz=timezone.utc)
    except Exception:
        return None
    return None


def backfill_reddit(video: Video) -> datetime | None:
    """Fetch publication timestamp from Reddit API."""
    try:
        # Reddit API: /api/info?id=t3_<source_id>
        resp = requests.get(
            f"https://www.reddit.com/api/info.json?id=t3_{video.source_id}",
            headers={"User-Agent": "RollingForge/0.1 backfill"},
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        children = resp.json().get("data", {}).get("children", [])
        if not children:
            return None
        created = children[0].get("data", {}).get("created_utc")
        if created:
            return datetime.fromtimestamp(float(created), tz=timezone.utc)
    except Exception:
        return None
    return None


def backfill_youtube(video: Video) -> datetime | None:
    """Fetch upload_date via yt-dlp flat extraction."""
    try:
        import yt_dlp  # lazy import
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video.source_url, download=False)
            upload = info.get("upload_date")  # YYYYMMDD
            if upload and len(upload) == 8:
                return datetime(
                    int(upload[:4]), int(upload[4:6]), int(upload[6:8]),
                    tzinfo=timezone.utc,
                )
    except Exception:
        return None
    return None


BACKFILLERS = {
    "bilibili": backfill_bilibili,
    "reddit": backfill_reddit,
    "youtube": backfill_youtube,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill videos.published_at.")
    parser.add_argument("--platform", choices=list(BACKFILLERS.keys()) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_migrations(get_settings().db_path)

    db = SessionLocal()
    try:
        platforms = [args.platform] if args.platform != "all" else list(BACKFILLERS.keys())
        total_updated = 0
        total_failed = 0

        for platform in platforms:
            q = (
                db.query(Video)
                .filter(Video.source_platform == platform)
                .filter(Video.published_at.is_(None))
                .order_by(Video.crawled_at.desc())
            )
            if args.limit:
                q = q.limit(args.limit)
            pending = q.all()

            logger.info("backfill %s: %d videos with NULL published_at", platform, len(pending))
            if not pending:
                continue

            backfiller = BACKFILLERS[platform]
            ok = 0
            failed = 0

            for i, v in enumerate(pending, 1):
                pub = backfiller(v)
                if pub:
                    if not args.dry_run:
                        v.published_at = pub
                        db.commit()
                    ok += 1
                    if i % 20 == 0:
                        logger.info("  %s: %d/%d done", platform, i, len(pending))
                else:
                    failed += 1
                # Rate limit — be nice to the APIs
                time.sleep(0.1)

            logger.info("%s done: ok=%d failed=%d", platform, ok, failed)
            total_updated += ok
            total_failed += failed

        logger.info("TOTAL: updated=%d failed=%d (dry_run=%s)", total_updated, total_failed, args.dry_run)
    finally:
        db.close()


if __name__ == "__main__":
    main()
