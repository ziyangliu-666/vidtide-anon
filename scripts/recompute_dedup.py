#!/usr/bin/env python3
"""Re-run dedup over existing DB rows, without re-crawling.

Use when:
- The captioner or embedder model has been upgraded and existing
  `caption_text` / `caption_model` / `duplicate_of_id` values are now
  stale (incompatible generation).
- You want to retune `cosine_threshold` and see how cluster assignments
  shift.
- You want to dedup a portion of the DB that was crawled before the
  dedup pipeline existed.

Behavior:
1. Clears `caption_text`, `caption_model`, `duplicate_of_id` on the
   target row set.
2. Clears the `vec_thumbnails` virtual table entries for the same rows.
3. Runs `dedupe_batch()` over them in chronological order, so canonical-
   choice ordering matches a normal pipeline run.

IMPORTANT: This script does NOT re-fetch thumbnails. It reads from the
local thumbnail cache (data/thumbnails/) and the `thumbnail_url` field.
Rows that don't have accessible thumbnails (e.g. showcase videos where
the thumbnail only existed transiently at crawl time) will be captioned
with an empty string — they were already in that state.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Project root on path for direct execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import sqlite_vec  # noqa: E402

from server.config import get_settings  # noqa: E402
from server.db.database import SessionLocal  # noqa: E402
from server.db.models import Video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute dedup over existing rows.")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process.")
    parser.add_argument(
        "--embedder",
        default="clip",
        choices=["clip"],
        help="Image embedder to use (default: clip = CLIP ViT-B/32).",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold (default: 0.85).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("recompute_dedup")

    settings = get_settings()
    db_path = settings.db_path

    db = SessionLocal()
    try:
        q = db.query(Video).order_by(Video.crawled_at.asc())
        if args.limit:
            q = q.limit(args.limit)
        videos = q.all()
        if not videos:
            logger.warning("No videos in DB, nothing to do.")
            return

        if not args.yes:
            print(f"About to recompute dedup on {len(videos)} videos.")
            print(f"  captioner: {args.captioner}")
            print(f"  cosine_threshold: {args.cosine_threshold}")
            print("  This will clear caption_text, caption_model, duplicate_of_id")
            print("  on all selected rows and delete matching vec_thumbnails entries.")
            resp = input("Proceed? [y/N] ").strip().lower()
            if resp not in ("y", "yes"):
                print("aborted")
                return

        # Clear dedup state on these rows
        ids = [v.id for v in videos]
        logger.info("clearing dedup state on %d rows", len(ids))
        for v in videos:
            v.caption_text = None
            v.caption_model = None
            v.duplicate_of_id = None
        db.commit()

        # Clear the vec_thumbnails entries for these ids
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Make sure table exists (first-run case)
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_thumbnails'"
        ).fetchone()
        if exists:
            for vid in ids:
                conn.execute("DELETE FROM vec_thumbnails WHERE video_id = ?", (vid,))
            conn.commit()
        conn.close()

        # Now run the dedup batch
        from server.dedup.deduplicator import dedupe_batch
        from server.dedup.image_embedder import get_image_embedder

        stats = dedupe_batch(
            videos,
            db,
            db_path=db_path,
            thumbnails_root="data/thumbnails",
            image_embedder=get_image_embedder(args.embedder),
            cosine_threshold=args.cosine_threshold,
        )
        logger.info("done: %s", stats)

        # Print cluster summary
        dups = db.query(Video).filter(Video.duplicate_of_id.isnot(None)).count()
        canon = db.query(Video).filter(
            Video.caption_text.isnot(None),
            Video.duplicate_of_id.is_(None),
        ).count()
        print()
        print(f"canonicals:  {canon}")
        print(f"duplicates:  {dups}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
