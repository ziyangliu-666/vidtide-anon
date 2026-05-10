#!/usr/bin/env python3
"""Delete local mp4 files for videos that have already been published to HF.

Under the HF-only architecture, "aging" is not a 30-day window demotion
from hot to cold — it's a post-publish local cleanup. The rule is:

    For every Video row where:
      - blob_url starts with https://huggingface.co/...  (published)
      AND
      - a stale local file still exists under data/blobs/videos/<id>.mp4
    Delete the local file. Leave the DB row untouched (blob_url already
    points at the HF copy, that's our authoritative source now).

This is safe to run multiple times (idempotent — missing local files
are a no-op) and safe to run at any time (HF is the authoritative source
so we're not racing anyone). The intended schedule is weekly via
.github/workflows/age-videos.yml, but you can also run it manually on
the crawler host to reclaim disk after a publish.

There is no more 30-day window. The fact that a video's blob_url points
at HF is the only signal we need — if it's there, HF has a permanent
copy and the local one is redundant.

Blob_sha256 is preserved on the row regardless, so benchmark users can
still verify a re-fetched copy even if HF later loses the file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from server.config import get_settings  # noqa: E402
from server.db.database import SessionLocal  # noqa: E402
from server.db.migrations import run_migrations  # noqa: E402
from server.db.models import Video  # noqa: E402

logger = logging.getLogger(__name__)


def _local_path_for(video_id: str, blob_dir: str) -> Path:
    """Mirror of DownloadStage's output filename convention."""
    return Path(blob_dir) / "videos" / f"{video_id}.mp4"


def _is_hf_url(url: str | None) -> bool:
    if not url:
        return False
    return url.startswith("https://huggingface.co/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete local mp4 files for videos already published to HF.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without touching the filesystem.",
    )
    parser.add_argument(
        "--blob-dir",
        default=None,
        help="Override blob_dir (default: settings.blob_dir from config).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Stop after this many deletions (default: unlimited).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    run_migrations(settings.db_path)

    blob_dir = args.blob_dir or settings.blob_dir

    db = SessionLocal()
    try:
        # Pull candidates: videos with an HF-hosted blob_url that may
        # still have a stale local copy. We don't filter in SQL on
        # file existence (SQLite can't) — we walk the candidates in
        # Python and check each one.
        candidates = (
            db.query(Video)
            .filter(Video.blob_url.isnot(None))
            .filter(Video.blob_url.like("https://huggingface.co/%"))
            .all()
        )
        logger.info(
            "aging: %d HF-published videos in DB, scanning for stale local blobs under %s",
            len(candidates), blob_dir,
        )

        deleted = 0
        skipped = 0
        bytes_freed = 0
        for video in candidates:
            local_path = _local_path_for(video.id, blob_dir)
            if not local_path.exists():
                skipped += 1
                continue
            size = local_path.stat().st_size
            if args.dry_run:
                logger.info(
                    "DRY-RUN would delete %s (%.1f MB)",
                    local_path, size / 1e6,
                )
            else:
                try:
                    local_path.unlink()
                    logger.info(
                        "deleted %s (%.1f MB)",
                        local_path, size / 1e6,
                    )
                except OSError:
                    logger.warning("failed to delete %s", local_path, exc_info=True)
                    continue
            deleted += 1
            bytes_freed += size
            if args.max is not None and deleted >= args.max:
                logger.info("hit --max limit, stopping")
                break

        logger.info(
            "aging done: %d files %s, %d skipped (not on disk), %.1f MB reclaimed",
            deleted,
            "would be deleted" if args.dry_run else "deleted",
            skipped,
            bytes_freed / 1e6,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
