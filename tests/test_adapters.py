from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from adapters import run_any4d


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _prediction(base_x: float, flow_x: float, depth: float, pose_x: float) -> dict[str, np.ndarray]:
    points = np.zeros((1, 1, 1, 3), dtype=np.float64)
    points[..., 0] = base_x
    points_camera = points.copy()
    points_camera[..., 2] = depth
    flow = np.zeros_like(points)
    flow[..., 0] = flow_x
    return {"pts3d": points, "pts3d_cam": points_camera, "scene_flow": flow, "c2w": _pose(pose_x)}


def test_any4d_chunks_share_frame_zero_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(run_any4d, "_camera_poses", lambda predictions: np.stack([pred["c2w"] for pred in predictions]))
    monkeypatch.setattr(run_any4d, "query_pixels", lambda *_: np.array([[0.0, 0.0]]))

    first = {
        "pred1": _prediction(2.0, 0.0, 10.0, 2.0),
        "pred2": _prediction(0.0, 1.0, 11.0, 3.0),
        "pred3": _prediction(0.0, 2.0, 12.0, 4.0),
    }
    second = {
        # This deliberately differs from the first anchor; frame zero must
        # retain the first chunk's prediction while the target is normalized
        # by this chunk's own anchor.
        "pred1": _prediction(5.0, 0.0, 99.0, 5.0),
        "pred2": _prediction(0.0, 3.0, 13.0, 8.0),
    }
    record = SimpleNamespace(frame_indices=[0, 6, 12, 18])
    arrays = run_any4d._extract_chunks(
        iter(((first, [0, 1, 2]), (second, [0, 3]))),
        None,
        record,
        {"tracking", "depth", "pose"},
    )

    np.testing.assert_allclose(arrays["tracking_xyz"][:, 0, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(arrays["depth"][:, 0, 0], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_allclose(arrays["camera_c2w"][:, 0, 3], [0.0, 1.0, 2.0, 3.0])
