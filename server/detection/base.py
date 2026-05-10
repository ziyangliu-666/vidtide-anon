"""BaseDetector interface + DetectionResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DetectionResult:
    """Single-video detector output."""

    video_id: str
    score: float                    # AI-generated probability in [0, 1]
    label_pred: str                 # "fake" | "real"
    raw: dict | None = None         # model-specific intermediate data


class BaseDetector(ABC):
    """Interface every detector must implement.

    Lifecycle:
      1. __init__(device="cuda") — load model + move to device
      2. predict(frames) — frames is (N, T, H, W, C) uint8, return score in [0,1]
      3. close() — free GPU memory (optional)

    Standardized input: 8 frames × 224×224 × RGB uint8. Resize happens in
    the dataset loader, not here.
    """

    name: str                       # "clip_zero_shot" | "demamba" | ...
    expects_frames: int = 8         # most video detectors take 8 uniform frames

    @abstractmethod
    def __init__(self, device: str = "cuda", **kwargs) -> None: ...

    @abstractmethod
    def predict(self, frames: np.ndarray) -> float:
        """frames: (T, H, W, 3) uint8 RGB. Returns AI-prob in [0, 1]."""

    def predict_batch(self, batch_frames: np.ndarray) -> np.ndarray:
        """batch_frames: (N, T, H, W, 3). Default impl: loop over N."""
        return np.array([self.predict(batch_frames[i]) for i in range(len(batch_frames))])

    def close(self) -> None:
        pass
