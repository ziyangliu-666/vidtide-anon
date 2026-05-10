"""Frame extraction for detector input.

Given a video file path, extract N uniformly-sampled frames at the target
resolution. Uses ffmpeg directly (no decord/torchvision deps required for
the initial CLIP zero-shot path).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    num_frames: int = 8,
    resolution: int = 224,
) -> np.ndarray:
    """Extract N uniform frames from a video file.

    Returns (N, H, W, 3) uint8 RGB array. Raises if the video is
    unreadable or too short.

    Strategy: probe duration, sample N timestamps evenly, use ffmpeg to
    extract one frame per timestamp at target resolution. This avoids
    decord/PyAV dependency issues on the first eval pass.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    # 1. probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        raise ValueError(f"ffprobe failed to read duration from {video_path}")

    if duration < 0.1:
        raise ValueError(f"Video too short: {duration}s")

    # 2. pick timestamps uniformly (avoid 0s and last frame to skip intros)
    eps = min(0.5, duration * 0.1)
    timestamps = np.linspace(eps, duration - eps, num_frames)

    # 3. extract each frame via ffmpeg to a temp PNG, read back as numpy
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, t in enumerate(timestamps):
            out = tmp / f"f{i:02d}.png"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{t:.3f}", "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale={resolution}:{resolution}:force_original_aspect_ratio=increase,"
                       f"crop={resolution}:{resolution}",
                str(out),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback: repeat last successful frame if available
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                else:
                    raise
            img = Image.open(out).convert("RGB")
            frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames, axis=0)
