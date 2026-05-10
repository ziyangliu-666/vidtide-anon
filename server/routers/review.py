"""Review router -- human review workflow for curated videos."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import Review, Video

router = APIRouter(tags=["review"])


# ---- Request / Response schemas ---------------------------------------------


class DetectionScoreOut(BaseModel):
    id: str
    detector_name: str
    confidence: float
    inference_time_ms: int | None = None
    run_id: str | None = None
    created_at: str

    class Config:
        from_attributes = True


class VideoForReview(BaseModel):
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
    content_tags: list[str] | dict | None = None
    has_watermark: bool | None = None
    storage_path: str | None = None
    thumbnail_path: str | None = None
    thumbnail_url: str | None = None
    status: str
    detection_scores: list[DetectionScoreOut] = []

    class Config:
        from_attributes = True


class ReviewRequest(BaseModel):
    decision: str  # "approved" | "rejected"


class ReviewResponse(BaseModel):
    id: str
    video_id: str
    decision: str
    reviewed_at: str


# ---- Routes -----------------------------------------------------------------


@router.get("/review/next", response_model=VideoForReview)
def next_unreviewed(db: Session = Depends(get_db)) -> VideoForReview:
    """Return the next candidate video that has no Review record yet."""
    already_reviewed = db.query(Review.video_id).subquery()

    video = (
        db.query(Video)
        .filter(Video.status != "excluded")
        .filter(~Video.id.in_(db.query(already_reviewed.c.video_id)))
        .order_by(Video.created_at.asc())
        .first()
    )

    if not video:
        raise HTTPException(status_code=404, detail="No unreviewed videos remaining")

    scores = [
        DetectionScoreOut(
            id=s.id,
            detector_name=s.detector_name,
            confidence=s.confidence,
            inference_time_ms=s.inference_time_ms,
            run_id=s.run_id,
            created_at=s.created_at.isoformat(),
        )
        for s in video.detection_scores
    ]

    return VideoForReview(
        id=video.id,
        source_platform=video.source_platform,
        source_url=video.source_url,
        source_id=video.source_id,
        label=video.label,
        label_source=video.label_source,
        label_confidence=video.label_confidence,
        claimed_generator=video.claimed_generator,
        duration_sec=video.duration_sec,
        resolution_w=video.resolution_w,
        resolution_h=video.resolution_h,
        fps=video.fps,
        file_size_bytes=video.file_size_bytes,
        content_tags=json.loads(video.content_tags) if video.content_tags else None,
        has_watermark=video.has_watermark,
        storage_path=video.storage_path,
        thumbnail_path=video.thumbnail_path,
        thumbnail_url=video.thumbnail_url,
        status=video.status,
        detection_scores=scores,
    )


@router.post("/videos/{video_id}/review", response_model=ReviewResponse)
def submit_review(
    video_id: str,
    body: ReviewRequest,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if body.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="Decision must be 'approved' or 'rejected'")

    review = Review(
        video_id=video_id,
        decision=body.decision,
        reviewed_at=datetime.now(timezone.utc),
    )
    db.add(review)

    video.status = "reviewed"
    db.commit()
    db.refresh(review)

    return ReviewResponse(
        id=review.id,
        video_id=review.video_id,
        decision=review.decision,
        reviewed_at=review.reviewed_at.isoformat(),
    )
