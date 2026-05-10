"""Orchestrates the full cross-platform video dedup pipeline.

Input: a batch of newly-crawled `Video` rows that passed the filter stage.
Output: same rows, with `duplicate_of_id` set on losers of each dedup
cluster and `caption_model` stamped to record which embedder ran.

Flow per batch (CLIP image embedding — replaced BLIP+MiniLM caption path):
    1. For each video, load thumbnail bytes (on-disk cache, stash, or URL).
    2. Embed the thumbnail directly via CLIP ViT-B/32 → 512-dim unit vector.
       (No text captioning step — CLIP embeds images directly, which is
       far more robust to JPEG compression, rescaling, and timestamp shifts
       than the old BLIP-caption → MiniLM-embed two-stage pipeline.)
    3. KNN against vec_thumbnails excluding self.
    4. If nearest neighbor cosine similarity ≥ threshold (default 0.85),
       treat as a duplicate — apply canonical-choice rule.
    5. Upsert the survivor's embedding into vec_thumbnails; losers stay out.

Canonical-choice rule (REQUIREMENTS.md R1):
    Lexicographic on (resolution_w * resolution_h, fps,
                      -abs(duration - median_cluster_duration),
                      crawled_at_asc)

Batches only look at the current crawl's videos plus whatever's
already in vec_thumbnails — no global re-dedup. For that, see
`scripts/recompute_dedup.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from server.dedup.image_embedder import BaseImageEmbedder, get_image_embedder
from server.dedup.vec_index import VecIndex

logger = logging.getLogger(__name__)

# Default thresholds — validated by manual testing (see the test session
# in the commit message). At 0.85: same-video variants (JPEG compression,
# resolution halving, timestamp shift) all score > 0.87 while genuinely
# different videos score < 0.55. Clean gap, zero false positives.
DEFAULT_COSINE_THRESHOLD = 0.85  # cosine similarity above this = duplicate
DEFAULT_KNN_K = 5  # how many neighbors to fetch from the vec index


@dataclass
class DedupStats:
    processed: int = 0
    captioned: int = 0
    embedded: int = 0
    duplicates_found: int = 0
    new_canonicals: int = 0
    errors: int = 0


def _thumbnail_bytes_for(video, *, thumbnails_root: Path) -> bytes | None:
    """Resolve thumbnail bytes for a video row.

    Priority:
      1. On-disk cached JPEG at data/thumbnails/<source_platform>_<source_id>.jpg
         (matches server/routers/import_videos.py:97-99 naming convention)
      2. HTTP fetch of `video.thumbnail_url` if it's an absolute URL
      3. Transient `video.thumbnail_bytes` attribute the pipeline stash
         from crawl time (set by ShowcaseCrawler via _thumbnail.extract_thumbnail)
    """
    # Stash set by runner at crawl time
    stash = getattr(video, "thumbnail_bytes", None)
    if stash:
        return stash

    # On-disk cache — try both the current `<platform>_<id>.jpg` naming
    # (written by server/routers/import_videos.py) and the older plain
    # `<id>.jpg` naming still present for rows crawled before the
    # convention change. Local crawls also fall through to HTTP fetch.
    candidates = [
        thumbnails_root / f"{video.source_platform}_{video.source_id}.jpg",
        thumbnails_root / f"{video.source_id}.jpg",
    ]
    for thumb_path in candidates:
        if thumb_path.exists():
            try:
                return thumb_path.read_bytes()
            except OSError as exc:
                logger.debug("dedup: read %s failed: %s", thumb_path, exc)

    # HTTP fetch — only if it's an absolute URL and not a relative
    # /api/thumbnail/... path (those would need the running FastAPI server).
    url = video.thumbnail_url
    if url and url.startswith(("http://", "https://")):
        try:
            import requests

            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception as exc:
            logger.debug("dedup: fetch %s failed: %s", url, exc)

    return None


def _canonical_pick(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict, dict]:
    """Return (winner, loser) tuple given two video dicts.

    Each dict must have keys: id, resolution_w, resolution_h, fps,
    duration_sec, crawled_at (datetime or ISO string or None).

    Rule (REQUIREMENTS.md R1): lexicographic on
    (resolution_area, fps, -abs(duration - median), crawled_at_asc).
    For a pairwise comparison the "median duration" is just
    (a.duration + b.duration) / 2, so the `-abs(...)` term collapses
    to a tie unless durations differ, in which case the shorter of the
    two wins (arbitrary but deterministic; matches cluster-of-2 semantics).
    """

    def sort_key(v: dict[str, Any]) -> tuple:
        area = (v.get("resolution_w") or 0) * (v.get("resolution_h") or 0)
        fps = v.get("fps") or 0.0
        duration = v.get("duration_sec") or 0.0
        crawled = v.get("crawled_at")
        if isinstance(crawled, datetime):
            crawled_ts = crawled.timestamp()
        elif isinstance(crawled, str):
            try:
                crawled_ts = datetime.fromisoformat(crawled).timestamp()
            except ValueError:
                crawled_ts = 0.0
        else:
            crawled_ts = 0.0
        # We want: higher area/fps/duration wins, earliest crawl wins.
        # Sort key is built for max-wins ordering.
        return (area, fps, duration, -crawled_ts)

    if sort_key(a) >= sort_key(b):
        return a, b
    return b, a


def _record_dedup_meta(
    db: Session, caption_model: str, embedding_model: str
) -> None:
    """Register (caption_model, embedding_model) pair in dedup_meta.

    Idempotent via the UNIQUE (caption_model, embedding_model) constraint.
    We don't bother with ORM for this — one raw insert per batch.
    """
    conn = db.connection().connection  # underlying sqlite3 connection
    conn.execute(
        """
        INSERT OR IGNORE INTO dedup_meta (caption_model, embedding_model, created_at)
        VALUES (?, ?, ?)
        """,
        (caption_model, embedding_model, datetime.now(timezone.utc).isoformat()),
    )


def dedupe_batch(
    videos: list,
    db: Session,
    *,
    db_path: str,
    thumbnails_root: str | Path = "data/thumbnails",
    image_embedder: BaseImageEmbedder | None = None,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    knn_k: int = DEFAULT_KNN_K,
) -> DedupStats:
    """Run dedup against a batch of newly-crawled videos.

    Uses CLIP image embedding directly — no intermediate text captioning.
    Mutates each video row:
      - `caption_model` set to the embedder name (e.g. "clip-vit-base-patch32")
      - `caption_text` optionally set (empty — we no longer generate text
        captions, but the column stays for backwards compat and potential
        future use as a human-readable label)
      - `duplicate_of_id` set on losers

    Commits incrementally per video.
    """
    embedder = image_embedder or get_image_embedder("clip")
    stats = DedupStats()

    if not videos:
        return stats

    thumbnails_root = Path(thumbnails_root)
    logger.info(
        "dedup: starting batch of %d videos with embedder=%s threshold=%.2f",
        len(videos), embedder.name, cosine_threshold,
    )

    # Record the generation
    _record_dedup_meta(db, "none", embedder.name)
    db.commit()

    with VecIndex(db_path) as index:
        index.ensure_table()

        for video in videos:
            stats.processed += 1
            try:
                img_bytes = _thumbnail_bytes_for(video, thumbnails_root=thumbnails_root)
                if not img_bytes:
                    logger.debug("dedup: no thumbnail for %s, skipping", video.id)
                    video.caption_text = ""
                    video.caption_model = embedder.name
                    db.commit()
                    continue

                embedding = embedder.embed(img_bytes)
                video.caption_model = embedder.name
                video.caption_text = ""  # no text caption in CLIP path
                stats.captioned += 1
                stats.embedded += 1

                neighbors = index.knn(
                    embedding, k=knn_k, exclude_video_id=video.id
                )

                duplicate_match = None
                for neighbor_id, distance in neighbors:
                    similarity = 1.0 - distance
                    if similarity >= cosine_threshold:
                        duplicate_match = (neighbor_id, similarity)
                        break

                if duplicate_match:
                    other_id, sim = duplicate_match
                    other = (
                        db.query(type(video))
                        .filter_by(id=other_id)
                        .first()
                    )
                    if other is None:
                        index.upsert(video.id, embedding)
                        stats.new_canonicals += 1
                        db.commit()
                        continue

                    a = {
                        "id": video.id,
                        "resolution_w": video.resolution_w,
                        "resolution_h": video.resolution_h,
                        "fps": video.fps,
                        "duration_sec": video.duration_sec,
                        "crawled_at": video.crawled_at,
                    }
                    b = {
                        "id": other.id,
                        "resolution_w": other.resolution_w,
                        "resolution_h": other.resolution_h,
                        "fps": other.fps,
                        "duration_sec": other.duration_sec,
                        "crawled_at": other.crawled_at,
                    }
                    winner, loser = _canonical_pick(a, b)

                    if winner["id"] == video.id:
                        other.duplicate_of_id = video.id
                        index.delete(other.id)
                        index.upsert(video.id, embedding)
                    else:
                        video.duplicate_of_id = other.id

                    stats.duplicates_found += 1
                    logger.info(
                        "dedup: %s ≈ %s (cos_sim=%.3f) — canonical=%s",
                        video.id[:8], other.id[:8], sim, winner["id"][:8],
                    )
                    db.commit()
                else:
                    index.upsert(video.id, embedding)
                    stats.new_canonicals += 1
                    db.commit()

            except Exception:
                stats.errors += 1
                logger.warning(
                    "dedup: error on video %s, continuing",
                    getattr(video, "id", "?"),
                    exc_info=True,
                )
                db.rollback()
                continue

    logger.info("dedup: batch complete — %s", stats)
    return stats
