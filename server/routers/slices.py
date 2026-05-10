"""Benchmark slice router — monthly window freeze + HF publish.

Two-step release flow (REQUIREMENTS.md R3):

1. **Freeze**: `POST /api/slices/freeze-window` — pure DB op. Aggregates
   reviewed videos in a [window_start, window_end) window into a new
   BenchmarkSlice row. `published_at=NULL` until the publish step lands.
   No network, never fails for external reasons. Monthly cadence.

2. **Publish**: `POST /api/slices/{id}/publish` — separate action. Calls
   the HF Datasets publisher synchronously. HF API outages and dataset
   card edits are independent failure modes and must be retryable
   without re-freezing the slice.

Old non-windowed `/freeze` route is gone. If you need to freeze the
current month with defaults, call `/freeze-window` with no body.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import BenchmarkSlice, Review, SliceVideo, Video

router = APIRouter(tags=["slices"])


# ---- Schemas ----------------------------------------------------------------


class SliceOut(BaseModel):
    id: str
    version: str
    total_videos: int
    approved_videos: int
    rejected_videos: int
    platforms: dict[str, int] | None = None
    generators: dict[str, int] | None = None
    created_at: str
    notes: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    published_at: str | None = None
    export_url: str | None = None

    class Config:
        from_attributes = True


class FreezeWindowRequest(BaseModel):
    version: str | None = None
    notes: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None


class FreezeResponse(BaseModel):
    id: str
    version: str
    total_videos: int
    approved_videos: int
    window_start: str | None
    window_end: str | None


class PublishResponse(BaseModel):
    id: str
    version: str
    export_url: str
    published_at: str


# ---- Helpers ----------------------------------------------------------------


def _slice_to_out(s: BenchmarkSlice) -> SliceOut:
    return SliceOut(
        id=s.id,
        version=s.version,
        total_videos=s.total_videos,
        approved_videos=s.approved_videos,
        rejected_videos=s.rejected_videos,
        platforms=json.loads(s.platforms) if s.platforms else None,
        generators=json.loads(s.generators) if s.generators else None,
        created_at=s.created_at.isoformat(),
        notes=s.notes,
        window_start=s.window_start.isoformat() if s.window_start else None,
        window_end=s.window_end.isoformat() if s.window_end else None,
        published_at=s.published_at.isoformat() if s.published_at else None,
        export_url=s.export_url,
    )


def _default_window_for(now: datetime) -> tuple[datetime, datetime]:
    """Return the month window containing `now` as a [start, end) pair.

    Start = first moment of this month, UTC. End = first moment of next
    month, UTC. Keeps slices non-overlapping and matches the monthly
    cadence contract in REQUIREMENTS.md R3.
    """
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if now.month == 12:
        end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _version_for_window(window_start: datetime) -> str:
    return f"vidtide-v{window_start.strftime('%Y.%m')}"


# ---- Routes -----------------------------------------------------------------


@router.get("/slices", response_model=list[SliceOut])
def list_slices(db: Session = Depends(get_db)) -> list[SliceOut]:
    slices = db.query(BenchmarkSlice).order_by(BenchmarkSlice.created_at.desc()).all()
    return [_slice_to_out(s) for s in slices]


@router.post("/slices/freeze-window", response_model=FreezeResponse)
def freeze_window(
    body: FreezeWindowRequest,
    db: Session = Depends(get_db),
) -> FreezeResponse:
    """Freeze reviewed videos in a window into a new monthly slice.

    If `window_start` / `window_end` are omitted, defaults to the
    current UTC month. Videos outside the window are ignored, even if
    reviewed — they belong to other slices. Only rows with status
    'reviewed' and an actual `Review` record count.

    Canonical videos only — rows with `duplicate_of_id IS NOT NULL`
    are excluded (their canonical survivor will be in the slice).

    Raises 400 if no videos in the window, 409 if a slice with the
    same `version` already exists.
    """
    now = datetime.now(timezone.utc)
    if body.window_start and body.window_end:
        window_start = body.window_start
        window_end = body.window_end
    else:
        window_start, window_end = _default_window_for(now)

    if window_end <= window_start:
        raise HTTPException(status_code=422, detail="window_end must be after window_start")

    version = body.version or _version_for_window(window_start)

    existing = db.query(BenchmarkSlice).filter(BenchmarkSlice.version == version).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Slice version '{version}' already exists",
        )

    # Build the review + video join, filtered to this window + canonical + reviewed.
    # Filter by published_at (platform upload date) NOT crawled_at — a slice
    # represents "videos published this month", not "videos we scraped this
    # month". Falls back to crawled_at for rows missing published_at (legacy
    # data before migration 0006).
    from sqlalchemy import or_, and_
    reviewed = (
        db.query(Video, Review)
        .join(Review, Review.video_id == Video.id)
        .filter(Video.status == "reviewed")
        .filter(Video.duplicate_of_id.is_(None))
        .filter(
            or_(
                and_(
                    Video.published_at.isnot(None),
                    Video.published_at >= window_start,
                    Video.published_at < window_end,
                ),
                and_(
                    Video.published_at.is_(None),
                    Video.crawled_at >= window_start,
                    Video.crawled_at < window_end,
                ),
            )
        )
        .all()
    )
    if not reviewed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No reviewed canonical videos in [{window_start.isoformat()}, "
                f"{window_end.isoformat()})"
            ),
        )

    approved_count = sum(1 for _, r in reviewed if r.decision == "approved")
    rejected_count = sum(1 for _, r in reviewed if r.decision == "rejected")

    platform_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    for v, r in reviewed:
        if r.decision == "approved":
            platform_counts[v.source_platform] = platform_counts.get(v.source_platform, 0) + 1
            gen = v.claimed_generator or "unknown"
            generator_counts[gen] = generator_counts.get(gen, 0) + 1

    bs = BenchmarkSlice(
        version=version,
        total_videos=len(reviewed),
        approved_videos=approved_count,
        rejected_videos=rejected_count,
        platforms=json.dumps(platform_counts),
        generators=json.dumps(generator_counts),
        notes=body.notes,
        window_start=window_start,
        window_end=window_end,
    )
    db.add(bs)
    db.flush()

    for video, review in reviewed:
        db.add(
            SliceVideo(
                slice_id=bs.id,
                video_id=video.id,
                decision=review.decision,
            )
        )
        video.status = "sliced"

    db.commit()

    return FreezeResponse(
        id=bs.id,
        version=version,
        total_videos=len(reviewed),
        approved_videos=approved_count,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )


@router.post("/slices/{slice_id}/publish", response_model=PublishResponse)
def publish(
    slice_id: str,
    db: Session = Depends(get_db),
) -> PublishResponse:
    """Push a frozen slice to HuggingFace Datasets.

    Synchronous — the manifest is small (a few KB to a few hundred KB),
    no need for a queue. Failure raises 500 with a readable detail so
    the caller can retry without re-freezing.

    Requires HF_TOKEN env var. Repo defaults to `vidtide/benchmark`;
    override with VIDTIDE_HF_REPO env var.
    """
    from server.release.hf_publisher import publish_slice

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN env var is not set",
        )
    repo_id = os.environ.get("VIDTIDE_HF_REPO", "vidtide/benchmark")

    bs = db.query(BenchmarkSlice).filter_by(id=slice_id).first()
    if not bs:
        raise HTTPException(status_code=404, detail="Slice not found")

    try:
        url = publish_slice(slice_id, db, hf_token=hf_token, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"publish failed: {exc}") from exc

    db.refresh(bs)
    return PublishResponse(
        id=bs.id,
        version=bs.version,
        export_url=url,
        published_at=bs.published_at.isoformat() if bs.published_at else "",
    )


@router.get("/slices/{slice_id}/export")
def export_slice(slice_id: str, db: Session = Depends(get_db)):
    """Return the full manifest for a slice as JSON (same schema as HF)."""
    from server.release.hf_publisher import build_manifest

    bs = db.query(BenchmarkSlice).filter(BenchmarkSlice.id == slice_id).first()
    if not bs:
        raise HTTPException(status_code=404, detail="Slice not found")
    return build_manifest(slice_id, db)
