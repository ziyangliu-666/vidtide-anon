"""Detector adapters used in paper Table 1.

Each adapter wraps the upstream detector's official inference code so that
``scripts/compute_gap.py`` can call them through a uniform interface. We do
**not** redistribute upstream weights; each adapter fetches its checkpoint
from the original GitHub release on first use.

Available detectors: see ``server/detection/detectors/`` for the full list.
"""
from server.detection.registry import get_detector, list_detectors

__all__ = ["get_detector", "list_detectors"]
