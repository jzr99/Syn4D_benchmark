"""Small geometry routines with no model-specific dependencies."""

from __future__ import annotations

import numpy as np


def unreal_camera_c2w(yaw: float, pitch: float, roll: float, xyz_cm: np.ndarray) -> np.ndarray:
    """Reproduce the Syn4D loader's Unreal-camera to OpenCV-world transform."""
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    r_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]],
        dtype=np.float64,
    )
    r_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    unreal_to_opencv = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_yaw @ r_pitch @ r_roll @ unreal_to_opencv
    out[:3, 3] = np.asarray(xyz_cm, dtype=np.float64) / 100.0
    return out


def relative_to_first(c2w: np.ndarray) -> np.ndarray:
    poses = np.asarray(c2w, dtype=np.float64)
    return np.linalg.inv(poses[0])[None] @ poses


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping row-vector points ``src`` to ``dst``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("Umeyama alignment needs matching [N,3] arrays with N >= 3")
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src0, dst0 = src - mu_src, dst - mu_dst
    covariance = (src0.T @ dst0) / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        sign[-1] = -1
    rotation = vt.T @ np.diag(sign) @ u.T
    variance = float(np.mean(np.sum(src0 * src0, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Degenerate source trajectory")
    scale = float(np.sum(singular * sign) / variance)
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def rotation_error_degrees(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(np.asarray(pred, dtype=np.float64), -1, -2) @ np.asarray(gt, dtype=np.float64)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
