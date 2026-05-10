"""SQLAlchemy ORM models for RollingForge."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from server.db.database import Base

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class Video(Base):
    __tablename__ = "videos"

    id = Column(String(32), primary_key=True, default=_uuid)
    source_platform = Column(String(32), nullable=False, index=True)
    source_url = Column(Text, nullable=False)
    source_id = Column(String(128), nullable=False)
    title = Column(Text, nullable=True)
    label = Column(String(16), nullable=False, default="unknown", index=True)  # real / fake / unknown
    label_source = Column(String(32), nullable=True)  # tier1_gallery | tier2_platform_tag | tier2_channel_whitelist | tier3_llm
    label_confidence = Column(Float, nullable=True)
    claimed_generator = Column(String(64), nullable=True)
    duration_sec = Column(Float, nullable=True)
    resolution_w = Column(Integer, nullable=True)
    resolution_h = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    content_tags = Column(Text, nullable=True)  # stored as JSON string
    # Cross-platform dedup fields (migration 0002). The `caption_text` comes
    # from a local img2txt model (Moondream2 by default) run against the
    # thumbnail; `caption_model` records which model version produced it so
    # lookups can refuse to compare cross-generation embeddings. The vector
    # itself lives in the sqlite-vec `vec_thumbnails` virtual table keyed by
    # video_id — single source of truth, no BLOB on this row.
    caption_text = Column(Text, nullable=True)
    caption_model = Column(String(64), nullable=True)
    # NULL = canonical survivor of its dedup cluster. Non-NULL = this video
    # is a duplicate of the referenced canonical video_id. Clusters are
    # derived as the transitive closure of these pointers via recursive CTE
    # in the dedup route handlers — no explicit cluster_id column.
    duplicate_of_id = Column(String(32), ForeignKey("videos.id"), nullable=True, index=True)
    # Blob location (migrations 0003 + 0005). `blob_url` fully describes
    # where the mp4 lives right now:
    #   NULL                    → not yet downloaded
    #   file://...              → local staging copy on the crawler host
    #   https://huggingface.co/ → published, authoritative copy on HF
    # `blob_sha256` is the reproducibility anchor — even if HF later
    # loses the file, researchers can verify a re-fetched copy.
    blob_url = Column(Text, nullable=True)
    blob_sha256 = Column(String(64), nullable=True)
    source_license = Column(String(32), nullable=True, default="unknown")
    has_watermark = Column(Boolean, nullable=True)
    storage_path = Column(Text, nullable=True)
    thumbnail_path = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="filtered", index=True)  # filtered (kept) / reviewed / sliced / excluded
    # published_at: platform-reported upload timestamp (from search API).
    # Distinct from crawled_at (when WE scraped it). Used by freeze-window
    # for monthly slice selection and by crawlers for max-age filtering.
    published_at = Column(DateTime, nullable=True, index=True)
    crawled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    # Manual curation pin (migration 0011). Non-NULL puts the video into
    # showcase tier 0 (above generator-tiered fakes), ordered by featured_at
    # DESC. Set via /videos/{id}/feature, cleared via /videos/{id}/unfeature.
    featured_at = Column(DateTime, nullable=True, index=True)

    detection_scores = relationship("DetectionScore", back_populates="video", lazy="selectin")
    reviews = relationship("Review", back_populates="video", lazy="selectin")

    __table_args__ = (
        Index("ix_videos_platform_label", "source_platform", "label"),
    )


# ---------------------------------------------------------------------------
# PipelineRun
# ---------------------------------------------------------------------------


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(String(32), primary_key=True, default=_uuid)
    run_name = Column(String(128), nullable=True)
    flow_type = Column(String(16), nullable=False)  # crawl / filter / detect / curate / full
    config = Column(Text, nullable=True)  # JSON
    status = Column(String(16), nullable=False, default="pending")  # pending / running / completed / failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    stats = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    detection_scores = relationship("DetectionScore", back_populates="run", lazy="selectin")


# ---------------------------------------------------------------------------
# DetectionScore
# ---------------------------------------------------------------------------


class DetectionScore(Base):
    __tablename__ = "detection_scores"

    id = Column(String(32), primary_key=True, default=_uuid)
    video_id = Column(String(32), ForeignKey("videos.id"), nullable=False, index=True)
    detector_name = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1
    inference_time_ms = Column(Integer, nullable=True)
    run_id = Column(String(32), ForeignKey("pipeline_runs.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    video = relationship("Video", back_populates="detection_scores")
    run = relationship("PipelineRun", back_populates="detection_scores")


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


class Review(Base):
    __tablename__ = "reviews"

    id = Column(String(32), primary_key=True, default=_uuid)
    video_id = Column(String(32), ForeignKey("videos.id"), nullable=False, index=True)
    decision = Column(String(16), nullable=False)  # approved / rejected
    reviewed_at = Column(DateTime, nullable=False, default=_utcnow)

    video = relationship("Video", back_populates="reviews")


# ---------------------------------------------------------------------------
# BenchmarkSlice
# ---------------------------------------------------------------------------


class BenchmarkSlice(Base):
    __tablename__ = "benchmark_slices"

    id = Column(String(32), primary_key=True, default=_uuid)
    version = Column(String(64), unique=True, nullable=False)  # vidtide-v2026.04
    total_videos = Column(Integer, nullable=False, default=0)
    approved_videos = Column(Integer, nullable=False, default=0)
    rejected_videos = Column(Integer, nullable=False, default=0)
    platforms = Column(Text, nullable=True)  # JSON summary
    generators = Column(Text, nullable=True)  # JSON summary
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    notes = Column(Text, nullable=True)
    # Monthly window semantics (migration 0004). Inclusive start, exclusive end.
    # freeze-window selects videos with crawled_at in [window_start, window_end).
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)
    # HF publish metadata. Set by server/release/hf_publisher.py after
    # upload_folder() lands. The original tier_mode column (hot/cold)
    # was dropped in migration 0005 when R2 went away — published slices
    # live on HF forever now.
    published_at = Column(DateTime, nullable=True)
    export_url = Column(Text, nullable=True)  # HF Datasets URL after publish


class SliceVideo(Base):
    __tablename__ = "slice_videos"

    slice_id = Column(String(32), ForeignKey("benchmark_slices.id"), primary_key=True)
    video_id = Column(String(32), ForeignKey("videos.id"), primary_key=True)
    decision = Column(String(16), nullable=False)  # approved / rejected
