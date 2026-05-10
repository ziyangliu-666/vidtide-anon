"""Detector registry — `@register_detector("name")` + `load_detector(name)`."""

from __future__ import annotations

import importlib
import logging
from typing import Callable, TypeVar

from server.detection.base import BaseDetector

logger = logging.getLogger(__name__)

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {}

T = TypeVar("T", bound=type[BaseDetector])


def register_detector(name: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        DETECTOR_REGISTRY[name] = cls
        return cls
    return deco


def load_detector(name: str, **kwargs) -> BaseDetector:
    if name not in DETECTOR_REGISTRY:
        # Lazy import the module — each detector file self-registers on import.
        try:
            importlib.import_module(f"server.detection.detectors.{name}")
        except ImportError as e:
            raise ValueError(f"Unknown detector '{name}': {e}") from e
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Detector '{name}' module imported but didn't register")
    return DETECTOR_REGISTRY[name](**kwargs)


def list_detectors() -> list[str]:
    return sorted(DETECTOR_REGISTRY.keys())
