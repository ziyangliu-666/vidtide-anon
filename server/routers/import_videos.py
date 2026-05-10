"""Import router — bulk import video metadata from local crawlers."""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import Video

# Persistent thumbnail cache lives on the Fly volume.
_DEFAULT_THUMB_DIR = Path("/app/data/thumbnails")
_LOCAL_THUMB_DIR = Path("data/thumbnails")
THUMB_DIR = _DEFAULT_THUMB_DIR if _DEFAULT_THUMB_DIR.parent.exists() else _LOCAL_THUMB_DIR
THUMB_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(tags=["import"])
logger = logging.getLogger(__name__)


def _require_api_key(x_api_key: str = Header(default="")):
    expected = os.environ.get("VIDTIDE_API_KEY", "")
    if not expected:
        # No key configured — allow all (development mode)
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


class VideoImportItem(BaseModel):
    source_platform: str
    source_url: str
    source_id: str
    title: str | None = None
    label: str = "fake"
    label_source: str | None = None
    claimed_generator: str | None = None
    duration_sec: float | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps: float | None = None
    content_tags: list[str] | None = None
    thumbnail_url: str | None = None
    # Locally-generated JPEG poster frame, base64-encoded. Set by
    # crawlers that can extract a poster (currently ShowcaseCrawler via
    # ffmpeg). When present, the import handler writes the bytes to the
    # Fly volume and overrides `thumbnail_url` to point at the cached
    # static-serve endpoint.
    thumbnail_b64: str | None = None
    status: str = "filtered"  # candidate pool; remote pipeline can override with "excluded" for pre-rejected rows
    # Dedup fields — pushed from crawler host after CLIP processing.
    # duplicate_of_source_id is the SOURCE_ID (not internal UUID) of the
    # canonical video this one is a duplicate of. The import handler
    # resolves it to the Fly-side UUID via (source_platform, source_id).
    caption_model: str | None = None
    duplicate_of_source_id: str | None = None  # source_id of canonical
    # Platform-reported publication timestamp (ISO 8601 string).
    # Distinct from crawled_at (when WE scraped). Used for month-window
    # slice selection and time-bounded crawling.
    published_at: str | None = None


class VideoImportRequest(BaseModel):
    videos: list[VideoImportItem]


class VideoImportResponse(BaseModel):
    imported: int
    updated: int = 0
    skipped: int = 0  # kept for backwards compat; always 0 under upsert semantics
    total: int


@router.post("/videos/import", response_model=VideoImportResponse)
def import_videos(
    body: VideoImportRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_require_api_key),
) -> VideoImportResponse:
    """Upsert videos by (source_platform, source_id).

    Existing rows have their crawl-time metadata refreshed (source_url, title,
    label, generator, dimensions, etc.) so re-pushes from the local crawler
    can correct mistakes — e.g. a showcase row that was originally pushed with
    a marketing-page URL gets the playable CDN URL on the next push. Fields
    that capture human-curation state (`status`, `crawled_at`) are preserved.
    """
    imported = 0
    updated = 0

    for item in body.videos:
        # Thumbnail handling:
        #
        # The crawler host writes thumbnails to data/thumbnails/<source_id>.jpg
        # and sets thumbnail_url to "/api/thumbnail/<source_id>.jpg" in the
        # local DB. When pushing to Fly, two cases:
        #
        # 1. thumbnail_b64 IS shipped: decode and write the bytes to the
        #    volume under the filename extracted from thumbnail_url (so
        #    the URL matches the file on disk). Preserves whatever naming
        #    convention the crawler host used.
        # 2. thumbnail_b64 NOT shipped: keep thumbnail_url as-is. If the
        #    file was written on a previous push, the URL still serves.
        #    If the URL is an absolute CDN URL (legacy rows), it serves
        #    from the platform CDN (may 403 on hotlink protection).
        thumbnail_url_override = item.thumbnail_url
        if item.thumbnail_b64:
            # Extract the filename from the incoming thumbnail_url when it's
            # a relative /api/thumbnail/* path. Fall back to the legacy
            # platform-prefixed convention for absolute URLs / legacy rows.
            if item.thumbnail_url and item.thumbnail_url.startswith("/api/thumbnail/"):
                thumb_name = item.thumbnail_url.split("/")[-1]
            else:
                thumb_name = f"{item.source_platform}_{item.source_id}.jpg"
            thumb_path = THUMB_DIR / thumb_name
            try:
                thumb_path.write_bytes(base64.b64decode(item.thumbnail_b64))
                thumbnail_url_override = f"/api/thumbnail/{thumb_name}"
            except (ValueError, OSError) as exc:
                logger.warning(
                    "Failed to persist thumbnail for %s/%s: %s",
                    item.source_platform,
                    item.source_id,
                    exc,
                )

        existing = db.query(Video).filter(
            Video.source_platform == item.source_platform,
            Video.source_id == item.source_id,
        ).first()

        # Resolve dedup: duplicate_of_source_id → Fly-side UUID
        dup_of_id = None
        if item.duplicate_of_source_id:
            canonical = db.query(Video).filter(
                Video.source_platform == item.source_platform,
                Video.source_id == item.duplicate_of_source_id,
            ).first()
            if canonical:
                dup_of_id = canonical.id

        # Parse published_at from ISO string into a datetime
        pub_dt = None
        if item.published_at:
            try:
                pub_dt = datetime.fromisoformat(item.published_at)
            except (ValueError, TypeError):
                pub_dt = None

        if existing is not None:
            existing.source_url = item.source_url
            existing.title = item.title
            existing.label = item.label
            existing.label_source = item.label_source
            existing.claimed_generator = item.claimed_generator
            existing.duration_sec = item.duration_sec
            existing.resolution_w = item.resolution_w
            existing.resolution_h = item.resolution_h
            existing.fps = item.fps
            existing.content_tags = (
                json.dumps(item.content_tags) if item.content_tags else None
            )
            existing.thumbnail_url = thumbnail_url_override
            if item.caption_model is not None:
                existing.caption_model = item.caption_model
            if dup_of_id is not None:
                existing.duplicate_of_id = dup_of_id
            if pub_dt is not None:
                existing.published_at = pub_dt
            # NOTE: deliberately NOT touching `status` (preserves human-review
            # decisions) or `crawled_at` (preserves first-seen timestamp).
            updated += 1
            continue

        video = Video(
            source_platform=item.source_platform,
            source_url=item.source_url,
            source_id=item.source_id,
            title=item.title,
            label=item.label,
            label_source=item.label_source,
            claimed_generator=item.claimed_generator,
            duration_sec=item.duration_sec,
            resolution_w=item.resolution_w,
            resolution_h=item.resolution_h,
            fps=item.fps,
            content_tags=json.dumps(item.content_tags) if item.content_tags else None,
            thumbnail_url=thumbnail_url_override,
            storage_path=None,
            thumbnail_path=None,
            caption_model=item.caption_model,
            duplicate_of_id=dup_of_id,
            published_at=pub_dt,
            status=item.status if item.status in ("filtered", "excluded") else "filtered",
            crawled_at=datetime.now(timezone.utc),
        )
        db.add(video)
        imported += 1

    db.commit()
    logger.info("Upserted videos: imported=%d updated=%d", imported, updated)

    return VideoImportResponse(
        imported=imported,
        updated=updated,
        skipped=0,
        total=len(body.videos),
    )
