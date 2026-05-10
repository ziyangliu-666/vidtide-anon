"""Local staging storage summary API.

Reports how many videos are still in local staging (file://...) vs
already published to HuggingFace (https://huggingface.co/...), plus
the size of the local staging directory. Powers the /storage page in
the dashboard.

Under the HF-only architecture there's no hot/cold tiering — every
video is either:
  - not yet downloaded (blob_url IS NULL)
  - locally staged (blob_url LIKE 'file://%')
  - published on HF (blob_url LIKE 'https://huggingface.co/%')
The aging job (`scripts/age_videos.py`) cleans up stale local files
for rows in the third state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import Video

router = APIRouter(tags=["storage"])


class StorageStatsResponse(BaseModel):
    total_videos: int
    not_downloaded: int   # canonical rows with blob_url IS NULL
    local_staging: int    # canonical rows with blob_url LIKE 'file://%'
    published: int        # canonical rows with blob_url LIKE 'https://huggingface.co/%'
    local_bytes: int      # SUM(file_size_bytes) for local_staging rows
    reclaimable: int      # published rows that still have a local_bytes entry


@router.get("/storage/stats", response_model=StorageStatsResponse)
def storage_stats(db: Session = Depends(get_db)) -> StorageStatsResponse:
    # Canonical rows only — duplicates don't need their own copy
    base = db.query(Video).filter(Video.duplicate_of_id.is_(None))

    total = base.count()

    not_downloaded = base.filter(Video.blob_url.is_(None)).count()

    local_staging = (
        base.filter(Video.blob_url.like("file://%")).count()
    )

    published = (
        base.filter(Video.blob_url.like("https://huggingface.co/%")).count()
    )

    local_bytes = (
        db.query(func.coalesce(func.sum(Video.file_size_bytes), 0))
        .filter(Video.duplicate_of_id.is_(None))
        .filter(Video.blob_url.like("file://%"))
        .scalar()
        or 0
    )

    # "Reclaimable" = published rows that still have a nonzero
    # file_size_bytes. Not a perfect signal (file_size_bytes survives
    # cleanup for reproducibility), but it's a heuristic of how much
    # the aging job would reclaim.
    reclaimable = (
        base.filter(Video.blob_url.like("https://huggingface.co/%"))
        .filter(Video.file_size_bytes.isnot(None))
        .filter(Video.file_size_bytes > 0)
        .count()
    )

    return StorageStatsResponse(
        total_videos=total,
        not_downloaded=not_downloaded,
        local_staging=local_staging,
        published=published,
        local_bytes=int(local_bytes),
        reclaimable=reclaimable,
    )
