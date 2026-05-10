#!/usr/bin/env python3
"""Stateless worker: pull pending URLs from Fly, download + dedup, report back.

This script treats the Fly.io DB as the single source of truth. It doesn't
need a local SQLite database — it pulls work items via the API, processes
them locally (yt-dlp + ffmpeg + CLIP), and PATCHes results back.

Flow:
  1. GET /api/videos/pending?limit=N
     → list of videos with source_url but no blob_url and no dedup
  2. For each video:
     a. Download mp4 via yt-dlp → re-encode to 720p → sha256
     b. CLIP embed the thumbnail → KNN search local vec index
     c. PATCH /api/videos/{id}/processed with blob_url + sha256 + dedup result
  3. The local vec index (sqlite-vec) is ephemeral — rebuilt each run from
     the thumbnails we download. It's a scratchpad, not a source of truth.

Usage:
  python scripts/process_pending.py --api-url https://your-vidtide-instance.example.com --limit 20
  python scripts/process_pending.py --api-url http://localhost:8000 --limit 50 --skip-download
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import requests

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


def _fetch_pending(api_url: str, api_key: str, limit: int) -> list[dict]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.get(
        f"{api_url}/api/videos/pending",
        params={"limit": limit},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _report_processed(api_url: str, api_key: str, video_id: str, body: dict) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.patch(
        f"{api_url}/api/videos/{video_id}/processed",
        json=body,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()


def _download_and_encode(source_url: str, video_id: str, work_dir: str) -> dict | None:
    """Download + re-encode to 720p. Returns {blob_path, blob_sha256, file_size_bytes} or None."""
    from server.pipeline.postprocess import DownloadStage
    from server.storage.local import LocalBlobStorage

    blob_dir = os.path.join(work_dir, "blobs")
    storage = LocalBlobStorage(blob_dir)
    stage = DownloadStage(storage, work_dir=os.path.join(work_dir, "tmp"))

    result = stage.process(source_url=source_url, video_id=video_id)
    if result is None:
        return None
    return {
        "blob_url": result.blob_url,
        "blob_sha256": result.blob_sha256,
        "file_size_bytes": result.file_size_bytes,
    }


def _clip_embed_thumbnail(thumbnail_url: str | None, api_url: str) -> list[float] | None:
    """Download thumbnail and compute CLIP embedding."""
    if not thumbnail_url:
        return None

    # Resolve thumbnail URL (may be relative /api/thumbnail/...)
    if thumbnail_url.startswith("/"):
        url = f"{api_url}{thumbnail_url}"
    else:
        url = thumbnail_url

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or not resp.content:
            return None
    except requests.RequestException:
        return None

    from server.dedup.image_embedder import ClipImageEmbedder

    embedder = _get_embedder()
    return embedder.embed(resp.content)


# Lazy-loaded CLIP embedder (one load per run)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from server.dedup.image_embedder import ClipImageEmbedder
        _embedder = ClipImageEmbedder()
    return _embedder


def _knn_search(embedding: list[float], vec_db_path: str, video_id: str, threshold: float = 0.85) -> str | None:
    """Search local ephemeral vec index for a near-duplicate. Returns canonical_id or None."""
    from server.dedup.vec_index import VecIndex

    with VecIndex(vec_db_path) as index:
        index.ensure_table()
        if index.count() == 0:
            # First video — just insert, no search
            index.upsert(video_id, embedding)
            return None

        neighbors = index.knn(embedding, k=1, exclude_video_id=video_id)
        # Insert into index (we always insert canonicals)
        if neighbors and (1.0 - neighbors[0][1]) >= threshold:
            # This video is a duplicate of an existing one
            return neighbors[0][0]
        else:
            # New canonical
            index.upsert(video_id, embedding)
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pending videos from Fly DB.")
    parser.add_argument("--api-url", default=os.environ.get("VIDTIDE_API_URL", "https://your-vidtide-instance.example.com"))
    parser.add_argument("--api-key", default=os.environ.get("VIDTIDE_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--work-dir", default="/tmp/vidtide-worker")
    parser.add_argument("--skip-download", action="store_true", help="Only run dedup, skip mp4 download")
    parser.add_argument("--skip-dedup", action="store_true", help="Only download, skip CLIP dedup")
    parser.add_argument("--threshold", type=float, default=0.85, help="CLIP cosine similarity threshold")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.makedirs(args.work_dir, exist_ok=True)
    # Ephemeral vec index for this run's dedup
    vec_db_path = os.path.join(args.work_dir, "ephemeral_vec.db")

    logger.info("fetching pending videos from %s (limit=%d)", args.api_url, args.limit)
    try:
        pending = _fetch_pending(args.api_url, args.api_key, args.limit)
    except requests.RequestException as exc:
        logger.error("failed to fetch pending: %s", exc)
        sys.exit(1)

    if not pending:
        logger.info("no pending videos")
        return

    logger.info("got %d pending videos", len(pending))

    ok = 0
    failed = 0
    dups = 0

    for item in pending:
        vid = item["id"]
        source_url = item["source_url"]
        logger.info("processing %s %s/%s", vid[:8], item["source_platform"], item["source_id"][:16])

        report: dict = {}

        # Step 1: Download
        if not args.skip_download:
            try:
                dl = _download_and_encode(source_url, vid, args.work_dir)
                if dl:
                    report.update(dl)
                    logger.info("  downloaded: %.1f MB sha=%s...", dl["file_size_bytes"] / 1e6, dl["blob_sha256"][:12])
                else:
                    logger.warning("  download failed, continuing")
            except Exception:
                logger.warning("  download crashed", exc_info=True)

        # Step 2: CLIP dedup
        if not args.skip_dedup:
            try:
                embedding = _clip_embed_thumbnail(item.get("thumbnail_url"), args.api_url)
                if embedding:
                    dup_of = _knn_search(embedding, vec_db_path, vid, args.threshold)
                    report["caption_model"] = "clip-vit-base-patch32"
                    if dup_of:
                        report["duplicate_of_id"] = dup_of
                        dups += 1
                        logger.info("  duplicate of %s", dup_of[:8])
                    else:
                        logger.info("  canonical (new)")
                else:
                    logger.info("  no thumbnail, skipping dedup")
            except Exception:
                logger.warning("  dedup crashed", exc_info=True)

        # Step 3: Report back
        if report:
            try:
                _report_processed(args.api_url, args.api_key, vid, report)
                ok += 1
            except requests.RequestException as exc:
                logger.warning("  report failed: %s", exc)
                failed += 1
        else:
            failed += 1

    logger.info("done: ok=%d failed=%d duplicates=%d", ok, failed, dups)

    # Cleanup ephemeral vec DB
    if os.path.exists(vec_db_path):
        os.unlink(vec_db_path)


if __name__ == "__main__":
    main()
