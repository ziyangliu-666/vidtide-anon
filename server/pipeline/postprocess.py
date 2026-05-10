"""Post-crawl pipeline stages that don't belong in the cloud FastAPI app.

The Fly.io cloud container is 512 MB and has OOM-killed ffmpeg before
(see `server/crawler/_thumbnail.py` header comment). Downloading and
re-encoding mp4s at ~20-100 MB each is out of the question there.

This module houses the post-crawl work that MUST run locally:

- `DownloadStage` — download a video via yt-dlp, re-encode to 720p / 30MB
  cap with ffmpeg, compute sha256, upload to blob storage, return the
  blob URL + hash.

The local driver script is `scripts/download_and_upload.py`; tests or
one-off scripts can import `DownloadStage` directly.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from server.storage.base import BaseBlobStorage
from server.utils.video import download_video

logger = logging.getLogger(__name__)

# Hard cap on output blob size. 720p short clips fit comfortably; longer
# or higher-bitrate sources get bitrate-capped in the re-encode step.
MAX_BLOB_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB


@dataclass
class DownloadResult:
    blob_key: str
    blob_url: str
    blob_sha256: str
    file_size_bytes: int


def _reencode_to_720p(src_path: str, dst_path: str, timeout_sec: int = 300) -> bool:
    """Re-encode `src_path` to H.264 720p capped at ~25 Mbps.

    This both normalizes input for detectors and keeps blob sizes
    predictable. 720p ceiling preserves aspect ratio via `scale=-2:720`
    (the -2 rounds width to even). `crf 23` is a reasonable quality
    target; `maxrate 4M + bufsize 8M` keeps bitrate bounded so a long
    source doesn't blow past MAX_BLOB_SIZE_BYTES.
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", src_path,
        "-vf", "scale=-2:'min(720,ih)'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-maxrate", "4M", "-bufsize", "8M",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dst_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
        if proc.returncode != 0:
            logger.warning(
                "reencode: ffmpeg rc=%d for %s: %s",
                proc.returncode, src_path,
                proc.stderr.decode("utf-8", errors="replace")[:200],
            )
            return False
        return os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("reencode: ffmpeg failed for %s: %s", src_path, exc)
        return False


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class DownloadStage:
    """Download + re-encode + upload a video to blob storage.

    Usage:
        stage = DownloadStage(blob_storage, work_dir="/tmp/vidtide-dl")
        result = stage.process(source_url="https://...", video_id="abc123")
        if result:
            # result has blob_key, blob_url, blob_sha256, file_size_bytes
            ...

    Each `process()` call uses a fresh subdirectory under `work_dir` and
    cleans it up after the blob upload completes (or fails).
    """

    def __init__(
        self,
        blob_storage: BaseBlobStorage,
        work_dir: str | Path = "/tmp/vidtide-dl",
        max_duration: int = 600,
    ) -> None:
        self.blob_storage = blob_storage
        self.work_dir = Path(work_dir)
        self.max_duration = max_duration
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def process(self, source_url: str, video_id: str) -> DownloadResult | None:
        tmp = tempfile.mkdtemp(prefix=f"dl-{video_id[:8]}-", dir=self.work_dir)
        try:
            raw_path = download_video(
                source_url, tmp, max_duration=self.max_duration
            )
            if not raw_path:
                logger.warning("DownloadStage: download failed for %s", video_id)
                return None

            reencoded = os.path.join(tmp, f"{video_id}.mp4")
            if not _reencode_to_720p(raw_path, reencoded):
                logger.warning("DownloadStage: reencode failed for %s", video_id)
                return None

            size = os.path.getsize(reencoded)
            if size > MAX_BLOB_SIZE_BYTES:
                logger.warning(
                    "DownloadStage: %s reencoded to %d bytes > cap %d; skipping upload",
                    video_id, size, MAX_BLOB_SIZE_BYTES,
                )
                return None

            sha = _sha256_file(reencoded)
            with open(reencoded, "rb") as f:
                data = f.read()

            blob_key = f"videos/{video_id}.mp4"
            blob_url = self.blob_storage.put(
                blob_key, data, content_type="video/mp4"
            )

            return DownloadResult(
                blob_key=blob_key,
                blob_url=blob_url,
                blob_sha256=sha,
                file_size_bytes=size,
            )
        finally:
            try:
                shutil.rmtree(tmp)
            except OSError:
                logger.debug("DownloadStage: failed to clean %s", tmp)
