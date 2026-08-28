"""Canonical constants shared by preparation, inference, and scoring."""

from __future__ import annotations

TRACKING_THRESHOLDS = (0.1, 0.3, 0.5, 1.0)
TRACKING_DYNAMIC_WEIGHT = 0.5
TRACKING_DYNAMIC_THRESHOLD_METERS = 0.01
DEPTH_MIN_METERS = 1e-3
DEPTH_MAX_METERS = 300.0
DEFAULT_FRAME_INDICES = tuple(range(0, 192, 6))
