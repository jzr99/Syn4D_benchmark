from __future__ import annotations

import numpy as np

from syn4d_benchmark.metrics import depth_metrics, pose_metrics, tracking_metrics


def test_tracking_perfect_and_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(4, 5, 3))
    valid = np.ones((4, 5), dtype=bool)
    dynamic = np.array([False, True, False, True, False])
    perfect = tracking_metrics(gt, gt * 7.0, valid, dynamic)
    assert perfect["score"] == 1.0
    assert perfect["apd"] == 1.0
    assert perfect["dynamic_apd"] == 1.0


def test_tracking_invalid_rows_are_ignored() -> None:
    gt = np.ones((2, 2, 3), dtype=np.float64)
    pred = gt.copy()
    pred[1, 1] = 1e6
    valid = np.array([[True, True], [True, False]])
    result = tracking_metrics(gt, pred, valid, np.array([False, True]))
    assert result["score"] == 1.0


def test_depth_perfect_and_scale_invariant() -> None:
    gt = np.linspace(1, 10, 24).reshape(2, 3, 4)
    result = depth_metrics(gt, gt * 4.0, align="scale")
    assert result["abs_rel"] < 1e-12
    assert result["delta1"] == 1.0


def test_pose_similarity_invariant() -> None:
    gt = np.tile(np.eye(4), (5, 1, 1))
    gt[:, 0, 3] = np.arange(5)
    angle = np.radians(30)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pred = gt.copy()
    pred[:, :3, :3] = rot.T[None] @ gt[:, :3, :3]
    pred[:, :3, 3] = (rot.T @ ((gt[:, :3, 3] - np.array([2, 3, 4])) / 3.0).T).T
    result = pose_metrics(gt, pred)
    assert result["ate_rmse"] < 1e-10
    assert result["rotation_mean_deg"] < 1e-5
