"""Stats router -- aggregate counts and recent pipeline runs."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.db.models import PipelineRun, Video

router = APIRouter(tags=["stats"])


def _expand_umbrella_platform(platform: str, content_tags_json: str | None) -> str:
    """Resolve a row's display bucket for the dashboard `by_platform` chart.

    For umbrella platforms (showcase, social, ...), look in `content_tags`
    for a `<platform>:<source_key>` entry stamped by the crawler and return
    that source_key. For everything else (or rows missing the tag), return
    the raw `source_platform` value unchanged. The expansion is uniform —
    no platform-name special-cases live in here, so adding a new umbrella
    crawler is zero-touch from this side.
    """
    if not content_tags_json:
        return platform
    try:
        tags = json.loads(content_tags_json)
    except (json.JSONDecodeError, TypeError):
        return platform
    prefix = f"{platform}:"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag[len(prefix):]
    return platform


# ---- Response schemas -------------------------------------------------------


class PipelineRunBrief(BaseModel):
    id: str
    run_name: str | None = None
    flow_type: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    stats: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    total_videos: int
    by_platform: dict[str, int]
    by_label: dict[str, int]
    by_status: dict[str, int]
    by_publish_month: dict[str, int]  # {"2026-04": 291, "2026-03": 222, ...}
    # Cross-tabs for stacked dashboard charts. Keys of the outer dict are
    # the primary bucket (platform or month); inner dict is label -> count.
    by_platform_label: dict[str, dict[str, int]]
    by_publish_month_label: dict[str, dict[str, int]]
    by_generator: dict[str, int]
    recent_runs: list[PipelineRunBrief]


# ---- Route ------------------------------------------------------------------


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)) -> StatsResponse:
    total_videos = db.query(func.count(Video.id)).scalar() or 0

    # Umbrella platforms (showcase, social, etc.) collapse multiple distinct
    # vendor sources onto one source_platform value, which is uninformative
    # in the dashboard. Convention: such crawlers stamp a `<platform>:<source_key>`
    # tag into content_tags (e.g. `showcase:deepmind_veo`, `social:tiktok`).
    # We expand any row with a matching tag into the per-source bucket. Rows
    # without the tag fall back to the raw platform name. Mirrors
    # `displayPlatform()` on the frontend so the pie chart matches the
    # videos-list Platform column.
    # Single pass: pull (platform, label, content_tags, published_at) for
    # every row and build all aggregations client-side. Avoids N round-trips.
    by_platform: dict[str, int] = {}
    by_platform_label: dict[str, dict[str, int]] = {}
    by_publish_month: dict[str, int] = {}
    by_publish_month_label: dict[str, dict[str, int]] = {}

    for platform, label, content_tags_json, pub_at in db.query(
        Video.source_platform, Video.label, Video.content_tags, Video.published_at
    ).all():
        bucket = _expand_umbrella_platform(platform, content_tags_json)
        by_platform[bucket] = by_platform.get(bucket, 0) + 1
        by_platform_label.setdefault(bucket, {})
        by_platform_label[bucket][label] = by_platform_label[bucket].get(label, 0) + 1

        month_key = pub_at.strftime("%Y-%m") if pub_at is not None else "unknown"
        by_publish_month[month_key] = by_publish_month.get(month_key, 0) + 1
        by_publish_month_label.setdefault(month_key, {})
        by_publish_month_label[month_key][label] = (
            by_publish_month_label[month_key].get(label, 0) + 1
        )

    by_label: dict[str, int] = {}
    for label, cnt in db.query(Video.label, func.count(Video.id)).group_by(Video.label).all():
        by_label[label] = cnt

    by_status: dict[str, int] = {}
    for status, cnt in db.query(Video.status, func.count(Video.id)).group_by(Video.status).all():
        by_status[status] = cnt

    by_generator: dict[str, int] = {}
    for gen, cnt in (
        db.query(Video.claimed_generator, func.count(Video.id))
        .filter(Video.status != "excluded")
        .filter(Video.claimed_generator.isnot(None))
        .filter(Video.claimed_generator != "")
        .group_by(Video.claimed_generator)
        .all()
    ):
        by_generator[gen] = cnt

    runs = (
        db.query(PipelineRun)
        .order_by(PipelineRun.created_at.desc())
        .limit(5)
        .all()
    )

    recent_runs = [
        PipelineRunBrief(
            id=r.id,
            run_name=r.run_name,
            flow_type=r.flow_type,
            status=r.status,
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            stats=json.loads(r.stats) if r.stats else None,
        )
        for r in runs
    ]

    return StatsResponse(
        total_videos=total_videos,
        by_platform=by_platform,
        by_label=by_label,
        by_status=by_status,
        by_publish_month=by_publish_month,
        by_platform_label=by_platform_label,
        by_publish_month_label=by_publish_month_label,
        by_generator=by_generator,
        recent_runs=recent_runs,
    )
