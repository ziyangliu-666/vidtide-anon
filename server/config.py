"""Application configuration: YAML pipeline config + Pydantic settings."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "pipeline.yaml"


class AppSettings(BaseSettings):
    """Runtime settings for the FastAPI server."""

    db_path: str = "data/vidtide.db"
    video_dir: str = "data/videos"
    data_dir: str = "data"
    # Directory for crawler-host-local blob storage. All downloaded mp4s
    # live here until they get uploaded to HuggingFace on monthly publish
    # and then aged out (deleted) by scripts/age_videos.py.
    blob_dir: str = "data/blobs"

    class Config:
        env_prefix = "ROLLINGFORGE_"


@functools.lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load and cache the YAML pipeline config."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached application settings."""
    cfg = get_config()
    return AppSettings(
        db_path=cfg.get("db_path", "data/vidtide.db"),
        video_dir=cfg.get("video_dir", "data/videos"),
        data_dir=cfg.get("data_dir", "data"),
    )
