"""Similarity-aligned camera-trajectory metrics."""

from __future__ import annotations

import numpy as np

from ..geometry import relative_to_first, rotation_error_degrees, umeyama_similarity


def _translation_angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = norm > 1e-12
    out = np.full(norm.shape, np.nan, dtype=np.float64)
    out[valid] = np.degrees(np.arccos(np.clip(np.sum(a[valid] * b[valid], axis=-1) / norm[valid], -1, 1)))
    return out


def pose_metrics(gt_c2w: np.ndarray, pred_c2w: np.ndarray) -> dict[str, float]:
    """Score one camera trajectory after first-frame normalization and Sim(3)."""
    gt = np.asarray(gt_c2w, dtype=np.float64)
    pred = np.asarray(pred_c2w, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[1:] != (4, 4):
        raise ValueError(f"poses must match [T,4,4], got {gt.shape} and {pred.shape}")
    if gt.shape[0] < 3 or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("pose evaluation needs at least three finite poses")
    gt, pred = relative_to_first(gt), relative_to_first(pred)
    scale, global_rotation, translation = umeyama_similarity(pred[:, :3, 3], gt[:, :3, 3])
    aligned = pred.copy()
    aligned[:, :3, 3] = (scale * (global_rotation @ pred[:, :3, 3].T)).T + translation
    aligned[:, :3, :3] = global_rotation[None] @ pred[:, :3, :3]
    position_error = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=-1)
    rotation_error = rotation_error_degrees(aligned[:, :3, :3], gt[:, :3, :3])

    gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
    pred_rel = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    rpe_rotation = rotation_error_degrees(pred_rel[:, :3, :3], gt_rel[:, :3, :3])
    rpe_translation = np.linalg.norm(pred_rel[:, :3, 3] - gt_rel[:, :3, 3], axis=-1)
    translation_angle = _translation_angle_degrees(pred_rel[:, :3, 3], gt_rel[:, :3, 3])
    pair_error = np.maximum(rpe_rotation, np.nan_to_num(translation_angle, nan=180.0))
    return {
        "ate_rmse": float(np.sqrt(np.mean(position_error**2))),
        "position_mean": float(np.mean(position_error)),
        "position_median": float(np.median(position_error)),
        "rotation_mean_deg": float(np.mean(rotation_error)),
        "rotation_median_deg": float(np.median(rotation_error)),
        "rpe_translation_rmse": float(np.sqrt(np.mean(rpe_translation**2))),
        "rpe_rotation_mean_deg": float(np.mean(rpe_rotation)),
        "translation_direction_mean_deg": float(np.nanmean(translation_angle)),
        "auc_5deg": float(np.mean(pair_error <= 5.0)),
        "auc_10deg": float(np.mean(pair_error <= 10.0)),
        "auc_20deg": float(np.mean(pair_error <= 20.0)),
        "alignment_scale": float(scale),
        "num_frames": int(gt.shape[0]),
    }
