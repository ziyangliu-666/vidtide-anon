"""Shared poster-frame extraction for crawlers.

Crawlers that have a direct CDN video URL but no vendor-supplied thumbnail
(e.g. ShowcaseCrawler, future TikTok/X/Vimeo crawlers ingesting hot-linkable
mp4s) should call `extract_thumbnail(url)` and stash the resulting JPEG bytes
on `CrawledVideo.thumbnail_bytes`. The runner ships those bytes via the
remote-push payload, the cloud import handler writes them to the Fly volume,
and the static `/api/thumbnail/<filename>.jpg` endpoint serves them.

Why this lives in a shared module rather than inside ShowcaseCrawler:
- Future crawlers should get thumbnails for free with one import line
- Keeps ffmpeg invocation, timeout policy, and quality settings in one place
- Lets us tune one thing without touching every crawler

Why we let ffmpeg do its own HTTP via `-i URL` instead of pre-fetching with
`requests.get(Range=...)`:
- mp4 files routinely have the moov atom at the END (e.g. Runway's Gen-4
  marketing CDN) so a simple front-range fetch never reaches the index
- ffmpeg's libavformat HTTP layer handles HEAD + tail-fetch + data-fetch
  range requests correctly for any container format
- Removes ~80 lines of fetch + tempfile + cleanup logic

Why local-only (not in the cloud request hot path):
- The Fly machine is a 512 MB shared-cpu-2x and OOM-killed uvicorn the
  moment we tried to run ffmpeg there. Local crawlers have plenty of RAM.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = "RollingForge/0.1 (research crawler; thumbnail extractor)"
_FFMPEG_BIN = shutil.which("ffmpeg")


def fetch_thumbnail_bytes(
    url: str,
    *,
    referer: str | None = None,
    timeout_sec: int = 15,
    max_width: int = 400,
    jpeg_quality: int = 80,
) -> bytes | None:
    """Download a thumbnail JPEG/PNG, resize, and return compact JPEG bytes.

    Used for platforms that already expose a thumbnail URL but hotlink-block
    cross-origin image requests from the dashboard (notably Bilibili's
    i*.hdslb.com CDN, which returns 403 `deny by referer access rule` when
    the Referer header isn't `bilibili.com`). We fetch locally at crawl
    time with the right Referer, then ship the bytes via the same
    `thumbnail_bytes` path that `extract_thumbnail()` uses for ffmpeg
    poster frames. Downstream the cloud writes them to /app/data/thumbnails
    and the dashboard serves them from the same-origin static endpoint,
    bypassing the hotlink check entirely.

    The downsample step is critical for the remote-push payload size:
    Bilibili cover images are typically 1920x1080 at ~250-400 KB each.
    At 120 clips per crawl that's 40+ MB of JPEG bytes, which (after
    base64 encoding) blasts past the Next.js front-proxy's 10 MB request
    body limit on your-vidtide-instance.example.com. Resizing to 400px wide drops each clip
    to ~10-20 KB so an entire crawl's thumbnails fit in a single chunk.
    The dashboard only uses 64x40 table cells and 320x180 detail-page
    cards anyway — 400px wide is overkill for both and still looks sharp
    on HiDPI displays.

    Returns compact JPEG bytes on success, None on any failure (network
    error, non-2xx, empty body, decode error). Silent by design — callers
    treat None as "no thumbnail" and the dashboard shows a placeholder.
    """
    if not url:
        return None
    headers = {"User-Agent": _USER_AGENT}
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec, stream=False)
    except requests.RequestException as exc:
        logger.debug("fetch_thumbnail_bytes: network error for %s: %s", url, exc)
        return None
    if resp.status_code != 200 or not resp.content:
        logger.debug(
            "fetch_thumbnail_bytes: status=%d body_len=%d for %s",
            resp.status_code, len(resp.content or b""), url,
        )
        return None

    raw = resp.content

    # Best-effort resize via Pillow. If Pillow isn't available or the
    # image can't be decoded, fall back to the raw bytes (still better
    # than no thumbnail at all).
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return raw

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:
        logger.debug("fetch_thumbnail_bytes: decode failed for %s: %s", url, exc)
        return raw

    # Convert RGBA/P palette PNGs to RGB so the JPEG encoder is happy.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if img.width > max_width:
        new_h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_h), Image.LANCZOS)

    buf = BytesIO()
    try:
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as exc:
        logger.debug("fetch_thumbnail_bytes: encode failed for %s: %s", url, exc)
        return raw
    return buf.getvalue()

# Defaults — see extract_thumbnail() for rationale.
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_POSTER_TIME = "0.5"
DEFAULT_WIDTH = 320


def extract_thumbnail(
    url: str,
    *,
    poster_time: str = DEFAULT_POSTER_TIME,
    width: int = DEFAULT_WIDTH,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> bytes | None:
    """Pull a single poster frame from a remote video URL via ffmpeg.

    Returns JPEG bytes on success, None on any failure (ffmpeg missing,
    network error, decode error, subprocess timeout). Failure is silent
    by design — the caller treats None as "no thumbnail" and the
    dashboard shows a placeholder.

    Parameters
    ----------
    url : str
        Direct video CDN URL (mp4, webm, mov, m4v). Must be HTTP-addressable;
        ffmpeg's libavformat does its own HTTP fetching.
    poster_time : str
        Timestamp to seek to, e.g. "0.5" or "00:00:01". Most encoders have
        a black or blank frame 0; 0.5s gives a more meaningful poster.
    width : int
        Output JPEG width in pixels. Aspect ratio is preserved.
    timeout_sec : int
        ffmpeg subprocess timeout. ffmpeg fetches via libavformat HTTP, so
        this needs to cover HEAD + tail-fetch + data-fetch + decode for
        slow CDNs.
    """
    if _FFMPEG_BIN is None:
        # ffmpeg not installed — degrade gracefully so crawls don't crash.
        return None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [
                _FFMPEG_BIN,
                "-y",
                "-loglevel", "error",
                "-user_agent", _USER_AGENT,
                "-ss", poster_time,
                "-i", url,
                "-vframes", "1",
                "-vf", f"scale={width}:-1",
                "-q:v", "5",
                str(out_path),
            ],
            capture_output=True,
            timeout=timeout_sec,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("extract_thumbnail: ffmpeg failed for %s: %s", url, exc)
        out_path.unlink(missing_ok=True)
        return None

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        logger.debug(
            "extract_thumbnail: ffmpeg returned %d for %s: %s",
            proc.returncode,
            url,
            proc.stderr.decode("utf-8", errors="replace")[:200],
        )
        out_path.unlink(missing_ok=True)
        return None

    try:
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)
