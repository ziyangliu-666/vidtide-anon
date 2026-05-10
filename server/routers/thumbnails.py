"""Static thumbnail serving for showcase clips.

Thumbnails are pre-generated locally by `ShowcaseCrawler._generate_thumbnail`
(which can afford to run ffmpeg) and shipped to the cloud as base64 bytes
in the import payload. The cloud import handler writes them to
`/app/data/thumbnails/{source_platform}_{source_id}.jpg` and points
`Video.thumbnail_url` at this endpoint.

This handler does ZERO video decoding — that's deliberate. The Fly machine
is a 512 MB shared-cpu-2x and can't afford ffmpeg in the request hot path
(an earlier on-demand version OOM-killed uvicorn the first time it ran).
On-disk JPEGs only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["thumbnails"])

# Same path resolution as import_videos.py — see that module for the
# rationale on the local-fallback shim.
_DEFAULT_THUMB_DIR = Path("/app/data/thumbnails")
_LOCAL_THUMB_DIR = Path("data/thumbnails")
THUMB_DIR = _DEFAULT_THUMB_DIR if _DEFAULT_THUMB_DIR.parent.exists() else _LOCAL_THUMB_DIR
THUMB_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/thumbnail/{filename}")
def get_thumbnail(filename: str) -> FileResponse:
    # Defensive: reject anything that tries to escape the thumbnail dir.
    # Filenames coming from the import handler are always
    # `{source_platform}_{source_id}.jpg` — no slashes, no `..`.
    if "/" in filename or ".." in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    path = THUMB_DIR / filename
    try:
        # resolve() and verify it's still inside THUMB_DIR — belt and braces
        # against any path-traversal trick I haven't thought of.
        path.resolve().relative_to(THUMB_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename")

    if not path.exists() or path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="thumbnail not found")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
