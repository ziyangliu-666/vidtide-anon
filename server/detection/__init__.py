"""Detector evaluation framework for VidTide.

Modules:
  base.py       — BaseDetector + DetectionResult dataclass
  registry.py   — @register_detector decorator + load_detector()
  detectors/    — per-model implementations
  dataset.py    — frame loader (ffmpeg → 8x224x224 tensor)
  metrics.py    — AUROC, bACC, F1, per-group heatmaps
  runner.py     — orchestrate eval across videos, cache scores

The runner reads videos from the DB (filtered by label/platform/slice),
extracts frames via ffmpeg, runs the detector, and writes per-video
confidence scores to `results/<detector>/<benchmark>.jsonl`. Metrics
are computed separately from cached scores.
"""
