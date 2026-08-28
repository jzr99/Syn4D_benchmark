"""WorldTrack-style 3D point tracking metrics."""

from __future__ import annotations

import numpy as np

from ..protocol import TRACKING_DYNAMIC_WEIGHT, TRACKING_THRESHOLDS


def tracking_metrics(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    valid: np.ndarray,
    dynamic: np.ndarray,
) -> dict[str, float]:
    """Score one sequence using one global median scale for all subsets."""
    gt = np.asarray(gt_xyz, dtype=np.float64)
    pred = np.asarray(pred_xyz, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    dynamic_q = np.asarray(dynamic, dtype=bool).reshape(-1)
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(f"tracking arrays must match [T,Q,3], got {gt.shape} and {pred.shape}")
    if valid_mask.shape != gt.shape[:2] or dynamic_q.shape != (gt.shape[1],):
        raise ValueError("tracking validity/dynamic shapes do not match tracks")
    finite = valid_mask & np.isfinite(gt).all(-1) & np.isfinite(pred).all(-1)
    if not finite.any():
        raise ValueError("sequence has no finite valid tracking correspondences")
    gt_norm = np.linalg.norm(gt[finite], axis=-1)
    pred_norm = np.linalg.norm(pred[finite], axis=-1)
    pred_median = float(np.median(pred_norm))
    scale = float(np.median(gt_norm) / pred_median) if pred_median > 1e-12 else 0.0
    distances = np.linalg.norm(gt - scale * pred, axis=-1)

    def subset(prefix: str, mask: np.ndarray) -> dict[str, float]:
        values = distances[mask]
        if values.size == 0:
            return {}
        fractions = {f"{prefix}acc_{threshold:g}m": float(np.mean(values <= threshold)) for threshold in TRACKING_THRESHOLDS}
        fractions[f"{prefix}apd"] = float(np.mean(list(fractions.values())))
        fractions[f"{prefix}epe"] = float(np.mean(values))
        fractions[f"{prefix}count"] = int(values.size)
        return fractions

    result: dict[str, float] = {"scale": scale}
    result.update(subset("", finite))
    dynamic_mask = finite & dynamic_q[None, :]
    result.update(subset("dynamic_", dynamic_mask))
    result["dynamic_query_fraction"] = float(np.mean(dynamic_q))
    if "dynamic_apd" in result:
        result["score"] = float(
            (1.0 - TRACKING_DYNAMIC_WEIGHT) * result["apd"]
            + TRACKING_DYNAMIC_WEIGHT * result["dynamic_apd"]
        )
    else:
        result["score"] = float(result["apd"])
    return result
