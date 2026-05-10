import json
import logging
import os
import subprocess
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str, max_duration: int = 60) -> str | None:
    """Download a video via yt-dlp to *output_dir*.

    Selects the best mp4 stream up to 720p.  Returns the path to the
    downloaded file, or ``None`` on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    def _duration_filter(info_dict, *, incomplete=False):
        """Reject videos longer than max_duration. Unknown duration = pass.

        The string-form filter `duration <= N` rejects records where
        duration is None — but yt-dlp's generic extractor (used for
        direct CDN mp4 URLs from showcase pages) doesn't probe the file
        before download and reports duration=None, so the static filter
        wrongly skips every showcase video. A Python callable lets us
        be explicit about the unknown case.
        """
        duration = info_dict.get("duration")
        if duration is None:
            return None  # unknown — accept
        if duration > max_duration:
            return f"too long: {duration:.0f}s > {max_duration}s"
        return None  # accept

    ydl_opts = {
        # Prefer single-stream mp4; fallback to best+merge for Reddit DASH etc.
        "format": "best[height<=720][ext=mp4]/best[height<=720]/bestvideo[height<=720]+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "match_filter": _duration_filter,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                logger.warning("download_video: yt-dlp returned no info for %s", url)
                return None

            # Method 1: check requested_downloads (most reliable)
            downloads = info.get("requested_downloads") or []
            for dl in downloads:
                fpath = dl.get("filepath") or dl.get("filename")
                if fpath and os.path.isfile(fpath):
                    return fpath

            # Method 2: check the expected filename from template
            filename = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            mp4_path = base + ".mp4"
            if os.path.isfile(mp4_path):
                return mp4_path
            if os.path.isfile(filename):
                return filename

            # Method 3: use the actual video ID (may differ from URL ID for Reddit)
            actual_id = info.get("id", "")
            if actual_id:
                for ext in (".mp4", ".webm", ".mkv"):
                    candidate = os.path.join(output_dir, actual_id + ext)
                    if os.path.isfile(candidate):
                        return candidate

            logger.warning("download_video: no file found after download (%s)", mp4_path)
            return None
    except Exception:
        logger.warning("download_video: failed to download %s", url, exc_info=True)
        return None


def get_video_info(path: str) -> dict:
    """Use ffprobe to extract video metadata.

    Returns a dict with keys: duration, width, height, fps, codec, file_size.
    On failure returns an empty dict.
    """
    if not os.path.isfile(path):
        logger.warning("get_video_info: file does not exist: %s", path)
        return {}

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("get_video_info: ffprobe failed for %s: %s", path, result.stderr)
            return {}
        probe = json.loads(result.stdout)
    except Exception:
        logger.warning("get_video_info: error running ffprobe on %s", path, exc_info=True)
        return {}

    info: dict = {}

    # File size from os.
    try:
        info["file_size"] = os.path.getsize(path)
    except OSError:
        info["file_size"] = 0

    # Duration from format section.
    fmt = probe.get("format") or {}
    duration_str = fmt.get("duration")
    if duration_str is not None:
        try:
            info["duration"] = float(duration_str)
        except (ValueError, TypeError):
            pass

    # Find the first video stream for resolution, fps, and codec.
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            info["codec"] = stream.get("codec_name")

            # Parse fps from r_frame_rate (e.g. "30/1" or "30000/1001").
            r_frame_rate = stream.get("r_frame_rate", "")
            if "/" in r_frame_rate:
                try:
                    num, den = r_frame_rate.split("/")
                    info["fps"] = float(num) / float(den)
                except (ValueError, ZeroDivisionError):
                    pass
            elif r_frame_rate:
                try:
                    info["fps"] = float(r_frame_rate)
                except ValueError:
                    pass
            break  # use first video stream only

    return info


def extract_thumbnail(
    video_path: str,
    output_path: str,
    time_sec: float = 1.0,
) -> str | None:
    """Extract a single frame as a JPEG thumbnail using ffmpeg.

    Returns *output_path* on success, or ``None`` on failure.
    """
    if not os.path.isfile(video_path):
        logger.warning("extract_thumbnail: video does not exist: %s", video_path)
        return None

    # Ensure parent directory of output exists.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(time_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("extract_thumbnail: ffmpeg failed for %s: %s", video_path, result.stderr)
            return None
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        logger.warning("extract_thumbnail: output file missing or empty after ffmpeg")
        return None
    except Exception:
        logger.warning("extract_thumbnail: error extracting thumbnail from %s", video_path, exc_info=True)
        return None
