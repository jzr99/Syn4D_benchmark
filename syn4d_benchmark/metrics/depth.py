"""Scale-aligned monocular video-depth metrics."""

from __future__ import annotations

import numpy as np

from ..protocol import DEPTH_MAX_METERS, DEPTH_MIN_METERS


def depth_metrics(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    *,
    min_depth: float = DEPTH_MIN_METERS,
    max_depth: float = DEPTH_MAX_METERS,
    align: str = "scale",
) -> dict[str, float]:
    """Score one video with one alignment shared by every frame and pixel."""
    gt = np.asarray(gt_depth, dtype=np.float64)
    pred = np.asarray(pred_depth, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"depth arrays must have the same shape, got {gt.shape} and {pred.shape}")
    valid = (
        np.isfinite(gt)
        & np.isfinite(pred)
        & (gt > float(min_depth))
        & (gt <= float(max_depth))
        & (pred > 0)
    )
    if not valid.any():
        raise ValueError("sequence has no valid depth pixels")
    g, p = gt[valid], pred[valid]
    if align == "scale":
        scale = float(np.median(g) / max(float(np.median(p)), 1e-12))
        shift = 0.0
    elif align == "scale_shift":
        design = np.stack([p, np.ones_like(p)], axis=1)
        scale, shift = np.linalg.lstsq(design, g, rcond=None)[0].tolist()
    elif align == "metric":
        scale, shift = 1.0, 0.0
    else:
        raise ValueError(f"unknown depth alignment: {align}")
    p = np.clip(scale * p + shift, float(min_depth), float(max_depth))
    g = np.clip(g, float(min_depth), float(max_depth))
    diff = p - g
    ratio = np.maximum(g / p, p / g)
    log_diff = np.log(p) - np.log(g)
    return {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(diff * diff / g)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "rmse_log": float(np.sqrt(np.mean(log_diff * log_diff))),
        "silog": float(np.sqrt(max(0.0, np.mean(log_diff * log_diff) - np.mean(log_diff) ** 2)) * 100.0),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "scale": float(scale),
        "shift": float(shift),
        "valid_pixels": int(g.size),
    }
