"""Videos router -- paginated listing and single-video detail."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import Video

router = APIRouter(tags=["videos"])


# ---- Response schemas -------------------------------------------------------


class DetectionScoreOut(BaseModel):
    id: str
    detector_name: str
    confidence: float
    inference_time_ms: int | None = None
    run_id: str | None = None
    created_at: str

    class Config:
        from_attributes = True


class VideoOut(BaseModel):
    id: str
    source_platform: str
    source_url: str
    source_id: str
    label: str
    label_source: str | None = None
    label_confidence: float | None = None
    claimed_generator: str | None = None
    duration_sec: float | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps: float | None = None
    file_size_bytes: int | None = None
    caption_text: str | None = None
    caption_model: str | None = None
    duplicate_of_id: str | None = None
    content_tags: Any | None = None
    has_watermark: bool | None = None
    storage_path: str | None = None
    thumbnail_path: str | None = None
    thumbnail_url: str | None = None
    status: str
    published_at: str | None = None
    crawled_at: str | None = None
    created_at: str
    featured_at: str | None = None

    class Config:
        from_attributes = True


class VideoDetailOut(VideoOut):
    detection_scores: list[DetectionScoreOut] = []


class VideoListResponse(BaseModel):
    items: list[VideoOut]
    total: int
    page: int
    pages: int


# ---- Helpers ----------------------------------------------------------------


def _video_to_out(v: Video) -> VideoOut:
    return VideoOut(
        id=v.id,
        source_platform=v.source_platform,
        source_url=v.source_url,
        source_id=v.source_id,
        label=v.label,
        label_source=v.label_source,
        label_confidence=v.label_confidence,
        claimed_generator=v.claimed_generator,
        duration_sec=v.duration_sec,
        resolution_w=v.resolution_w,
        resolution_h=v.resolution_h,
        fps=v.fps,
        file_size_bytes=v.file_size_bytes,
        caption_text=v.caption_text,
        caption_model=v.caption_model,
        duplicate_of_id=v.duplicate_of_id,
        content_tags=json.loads(v.content_tags) if v.content_tags else None,
        has_watermark=v.has_watermark,
        storage_path=v.storage_path,
        thumbnail_path=v.thumbnail_path,
        thumbnail_url=v.thumbnail_url,
        status=v.status,
        published_at=v.published_at.isoformat() if v.published_at else None,
        crawled_at=v.crawled_at.isoformat() if v.crawled_at else None,
        created_at=v.created_at.isoformat(),
        featured_at=v.featured_at.isoformat() if v.featured_at else None,
    )


def _video_to_detail(v: Video) -> VideoDetailOut:
    scores = [
        DetectionScoreOut(
            id=s.id,
            detector_name=s.detector_name,
            confidence=s.confidence,
            inference_time_ms=s.inference_time_ms,
            run_id=s.run_id,
            created_at=s.created_at.isoformat(),
        )
        for s in v.detection_scores
    ]
    base = _video_to_out(v)
    return VideoDetailOut(**base.model_dump(), detection_scores=scores)


# ---- Routes -----------------------------------------------------------------


_SORT_COLUMNS = {
    "created_at": Video.created_at,
    "published_at": Video.published_at,
    "crawled_at": Video.crawled_at,
    "duration_sec": Video.duration_sec,
    "title": Video.title,
}


# Showcase sort: photorealistic SOTA fakes first, cartoon/old-gen last.
# Reviewers landing on /videos see the most convincing AI generations first.
_SHOWCASE_TOP_GENERATORS = {
    "sora2", "sora", "veo3", "kling21", "kling2",
    "runway-gen4", "dreamina3", "hailuo", "luma", "vidu",
    "wan21", "pixverse", "hunyuan", "pika2",
}
_SHOWCASE_BOTTOM_GENERATORS = {
    "runway_gen2", "runway-gen3", "veo2", "pika1",
    "stable_video_diffusion", "kling1", "ltxv",
    "animatediff", "modelscope_t2v",
}


_CARTOON_KEYWORDS_LOWER = ("anime", "cartoon", "waifu", "mmd")
_CARTOON_KEYWORDS_CN = (
    "动画", "动漫", "卡通", "二次元", "漫画",
    "国漫", "番剧", "手办", "萌宠",
)


def _to_unicode_escape(s: str) -> str:
    """`动画` -> `\\u52a8\\u753b`. content_tags is JSON-encoded with
    ensure_ascii=True, so Chinese characters live as escape sequences in the DB."""
    return "".join(f"\\u{ord(c):04x}" for c in s)


def _matches_cartoon_in_title(col):
    clauses = [func.lower(col).like(f"%{kw}%") for kw in _CARTOON_KEYWORDS_LOWER]
    clauses += [col.like(f"%{kw}%") for kw in _CARTOON_KEYWORDS_CN]
    return or_(*clauses)


def _matches_cartoon_in_tags(col):
    # content_tags is JSON-encoded with escaped unicode (e.g. `动画`),
    # so literal Chinese LIKE patterns don't match. Match the escape form.
    # Also match literal as a fallback in case any rows were inserted differently.
    clauses = [func.lower(col).like(f"%{kw}%") for kw in _CARTOON_KEYWORDS_LOWER]
    for kw in _CARTOON_KEYWORDS_CN:
        clauses.append(col.like(f"%{kw}%"))
        clauses.append(col.like(f"%{_to_unicode_escape(kw)}%"))
    return or_(*clauses)


def _showcase_tier_expr():
    # Tier 0: manually pinned via the admin button (featured_at NOT NULL).
    # Then: demote anime/cartoon to tier 3 EVEN IF the generator is top-tier
    # (Dreamina3 / Kling21 / Pika2 produce anime well — title-only filter
    # missed these because most fakes have empty titles; check content_tags too).
    cartoon_in_title = _matches_cartoon_in_title(Video.title)
    cartoon_in_tags = _matches_cartoon_in_tags(Video.content_tags)
    return case(
        (Video.featured_at.isnot(None), 0),
        (cartoon_in_title, 3),
        (cartoon_in_tags, 3),
        (Video.claimed_generator.in_(_SHOWCASE_TOP_GENERATORS), 1),
        (Video.claimed_generator.in_(_SHOWCASE_BOTTOM_GENERATORS), 3),
        else_=2,
    )


@router.get("/videos", response_model=VideoListResponse)
def list_videos(
    platform: str | None = Query(None),
    label: str | None = Query(None),
    status: str | None = Query(None),
    generator: str | None = Query(None),
    q: str | None = Query(None, description="Search title / source_url / source_id"),
    min_duration: float | None = Query(None, ge=0),
    max_duration: float | None = Query(None, ge=0),
    sort: str = Query("created_at"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> VideoListResponse:
    query = db.query(Video)
    if platform:
        query = query.filter(Video.source_platform == platform)
    if label:
        query = query.filter(Video.label == label)
    if status:
        query = query.filter(Video.status == status)
    else:
        query = query.filter(Video.status != "excluded")
    if generator:
        query = query.filter(Video.claimed_generator == generator)
    if min_duration is not None:
        query = query.filter(Video.duration_sec >= min_duration)
    if max_duration is not None:
        query = query.filter(Video.duration_sec <= max_duration)
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                Video.title.ilike(like),
                Video.source_url.ilike(like),
                Video.source_id.ilike(like),
            )
        )

    if sort == "showcase":
        # Tier ascending (0=featured, 1=premium, 3=cartoon/old). Within tier 0,
        # most recently pinned first (featured_at DESC). Within other tiers,
        # featured_at is NULL so it's a no-op and we fall through to published_at.
        tier_expr = _showcase_tier_expr()
        query = query.order_by(
            tier_expr.asc(),
            Video.featured_at.desc().nulls_last(),
            Video.published_at.desc().nulls_last(),
            Video.crawled_at.desc().nulls_last(),
            Video.id.asc(),
        )
    else:
        sort_col = _SORT_COLUMNS.get(sort, Video.created_at)
        sort_expr = sort_col.asc() if order == "asc" else sort_col.desc()
        # Stable tiebreaker so pagination doesn't drift for rows with equal sort keys.
        query = query.order_by(sort_expr, Video.id.asc())

    total = query.count()
    pages = max(1, math.ceil(total / per_page))
    items = query.offset((page - 1) * per_page).limit(per_page).all()

    return VideoListResponse(
        items=[_video_to_out(v) for v in items],
        total=total,
        page=page,
        pages=pages,
    )


# ---- Pending work + processed report ----------------------------------------
#
# Contract between Fly (single source of truth DB) and the crawler host
# (stateless worker). The crawler pulls a list of videos needing processing,
# does the heavy work (download + CLIP dedup) locally, then reports results
# back to Fly. No local SQLite needed on the crawler host.


class PendingVideoItem(BaseModel):
    id: str
    source_platform: str
    source_url: str
    source_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    duration_sec: float | None = None
    claimed_generator: str | None = None

    class Config:
        from_attributes = True


@router.get("/videos/pending", response_model=list[PendingVideoItem])
def list_pending(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[PendingVideoItem]:
    """Videos that need processing: filtered, no download, no dedup, canonical.

    The crawler host polls this, downloads each video, runs CLIP dedup,
    and PATCHes results back via /videos/{id}/processed.

    Order: oldest first so the worker makes steady forward progress.
    """
    rows = (
        db.query(Video)
        .filter(Video.status != "excluded")
        .filter(Video.blob_url.is_(None))
        .filter(Video.duplicate_of_id.is_(None))
        .filter(Video.caption_model.is_(None))
        .order_by(Video.crawled_at.asc())
        .limit(limit)
        .all()
    )
    return [
        PendingVideoItem(
            id=v.id,
            source_platform=v.source_platform,
            source_url=v.source_url,
            source_id=v.source_id,
            title=v.title,
            thumbnail_url=v.thumbnail_url,
            duration_sec=v.duration_sec,
            claimed_generator=v.claimed_generator,
        )
        for v in rows
    ]


class ProcessedReport(BaseModel):
    """Report from the crawler host after processing a video."""
    # Download result (optional — worker may skip download)
    blob_url: str | None = None
    blob_sha256: str | None = None
    file_size_bytes: int | None = None
    # Dedup result (optional — worker may skip dedup)
    caption_model: str | None = None
    duplicate_of_id: str | None = None  # NULL = canonical, non-NULL = points to canonical


class ProcessedResponse(BaseModel):
    id: str
    status: str


@router.patch("/videos/{video_id}/processed", response_model=ProcessedResponse)
def report_processed(
    video_id: str,
    body: ProcessedReport,
    db: Session = Depends(get_db),
) -> ProcessedResponse:
    """Crawler host reports processing results for a single video.

    Updates whichever fields are non-null in the body. Does NOT touch
    status or label — those are owned by the review workflow.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if body.blob_url is not None:
        video.blob_url = body.blob_url
    if body.blob_sha256 is not None:
        video.blob_sha256 = body.blob_sha256
    if body.file_size_bytes is not None:
        video.file_size_bytes = body.file_size_bytes
    if body.caption_model is not None:
        video.caption_model = body.caption_model
    if body.duplicate_of_id is not None:
        video.duplicate_of_id = body.duplicate_of_id
    db.commit()
    return ProcessedResponse(id=video.id, status=video.status)


@router.get("/videos/{video_id}", response_model=VideoDetailOut)
def get_video(video_id: str, db: Session = Depends(get_db)) -> VideoDetailOut:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video or video.status == "excluded":
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_to_detail(video)


# ---- Admin curation endpoints ----------------------------------------------
#
# Used by the temporary `/videos?admin=1` buttons during reviewer-prep
# curation. No auth — frontend hides buttons unless the URL flag is set.
# Remove these endpoints + buttons once curation is done.


class FeatureResponse(BaseModel):
    id: str
    featured_at: str | None


@router.post("/videos/{video_id}/feature", response_model=FeatureResponse)
def feature_video(video_id: str, db: Session = Depends(get_db)) -> FeatureResponse:
    """Pin to showcase tier 0 (top of list). Sets featured_at = now."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.featured_at = datetime.now(timezone.utc)
    db.commit()
    return FeatureResponse(id=video.id, featured_at=video.featured_at.isoformat())


@router.post("/videos/{video_id}/unfeature", response_model=FeatureResponse)
def unfeature_video(video_id: str, db: Session = Depends(get_db)) -> FeatureResponse:
    """Remove from tier 0; falls back to its natural showcase tier."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.featured_at = None
    db.commit()
    return FeatureResponse(id=video.id, featured_at=None)


class DeleteResponse(BaseModel):
    id: str
    deleted: bool


@router.delete("/videos/{video_id}", response_model=DeleteResponse)
def delete_video(video_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    """Hard delete the video and all FK-referencing rows."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    # Clear FK references that would prevent deletion. Other videos that
    # pointed to this one as their canonical become canonical themselves.
    db.execute(
        Video.__table__.update()
        .where(Video.duplicate_of_id == video_id)
        .values(duplicate_of_id=None)
    )
    from sqlalchemy import text
    for tbl in ("detection_scores", "reviews", "slice_videos"):
        db.execute(text(f"DELETE FROM {tbl} WHERE video_id = :vid"), {"vid": video_id})
    db.delete(video)
    db.commit()
    return DeleteResponse(id=video_id, deleted=True)
