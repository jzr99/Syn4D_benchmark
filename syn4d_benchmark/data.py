"""Ground-truth and prediction I/O."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .geometry import unreal_camera_c2w
from .manifest import SequenceRecord


TRACKING_GT_SCHEMA = "syn4d-tracking-gt-v1"
TRACKING_GT_ARRAY_FIELDS = (
    "tracks_ref0",
    "visibility",
    "dynamic",
    "query_uv",
    "frame_indices",
    "intrinsics",
    "image_size",
)


def prediction_path(root: Path, record: SequenceRecord) -> Path:
    return Path(root) / record.variant / record.scene / f"{record.sequence}.npz"


def tracking_gt_path(root: Path, record: SequenceRecord) -> Path:
    return Path(root) / record.variant / record.scene / f"{record.sequence}.npy"


def save_tracking_gt(path: Path, sequence_id: str, **arrays: np.ndarray) -> None:
    """Save one self-contained tracking pack as a non-pickled structured NPY."""
    missing = set(TRACKING_GT_ARRAY_FIELDS) - arrays.keys()
    extra = arrays.keys() - set(TRACKING_GT_ARRAY_FIELDS)
    if missing or extra:
        raise ValueError(f"tracking GT fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
    normalized = {name: np.asarray(arrays[name]) for name in TRACKING_GT_ARRAY_FIELDS}
    dtype = np.dtype(
        [("schema", "U32"), ("sequence_id", "U160")]
        + [(name, value.dtype, value.shape) for name, value in normalized.items()]
    )
    payload = np.zeros((), dtype=dtype)
    payload["schema"] = TRACKING_GT_SCHEMA
    payload["sequence_id"] = sequence_id
    for name, value in normalized.items():
        payload[name] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, payload, allow_pickle=False)
    temporary.replace(path)


def load_tracking_gt(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a self-contained tracking NPY without pickle support."""
    path = Path(path)
    payload = np.load(path, allow_pickle=False)
    names = set(payload.dtype.names or ())
    required = {"schema", "sequence_id", *TRACKING_GT_ARRAY_FIELDS}
    if payload.shape != () or not required.issubset(names):
        raise ValueError(f"{path} is not a {TRACKING_GT_SCHEMA} structured NPY")
    schema = str(payload["schema"].item())
    if schema != TRACKING_GT_SCHEMA:
        raise ValueError(f"Unsupported tracking GT schema {schema!r} in {path}")
    result = {name: np.asarray(payload[name]) for name in TRACKING_GT_ARRAY_FIELDS}
    tracks = result["tracks_ref0"]
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_ref0 must be [T,Q,3] in {path}, got {tracks.shape}")
    frames, queries = tracks.shape[:2]
    expected_shapes = {
        "visibility": (frames, queries),
        "dynamic": (queries,),
        "query_uv": (queries, 2),
        "frame_indices": (frames,),
        "intrinsics": (4,),
        "image_size": (2,),
    }
    for name, expected in expected_shapes.items():
        if result[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected} in {path}, got {result[name].shape}")
    result["sequence_id"] = np.asarray(str(payload["sequence_id"].item()))
    return result


def read_depth_exr(path: Path) -> np.ndarray:
    try:
        import Imath
        import OpenEXR

        handle = OpenEXR.InputFile(str(path))
        window = handle.header()["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channels = handle.header().get("channels", {})
        for name in ("Depth", "Z", "R", "Y"):
            if name in channels:
                raw = handle.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width) / 100.0
                handle.close()
                return depth
        handle.close()
        raise RuntimeError(f"No depth channel in {path}; found {list(channels)}")
    except ImportError as exc:
        raise RuntimeError("Reading Syn4D depth requires the OpenEXR and Imath Python packages") from exc


def load_depth_gt(record: SequenceRecord) -> np.ndarray:
    paths = [Path(record.depth_dir) / f"{record.sequence}_{frame:04d}_depth.exr" for frame in record.frame_indices]
    return np.stack([read_depth_exr(path) for path in paths]).astype(np.float32)


def load_camera_gt(record: SequenceRecord) -> np.ndarray:
    with Path(record.camera_csv).open(newline="", encoding="utf-8") as handle:
        rows = {Path(row["name"]).name: row for row in csv.DictReader(handle)}
    poses = []
    for frame in record.frame_indices:
        name = f"{record.sequence}_{frame:04d}.png"
        row = rows.get(name)
        if row is None:
            raise KeyError(f"Camera metadata has no row for {name}")
        poses.append(
            unreal_camera_c2w(
                float(row["yaw"]),
                float(row["pitch"]),
                float(row["roll"]),
                np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            )
        )
    return np.stack(poses).astype(np.float64)


def resize_depth_video(pred: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 3:
        raise ValueError(f"predicted depth must be [T,H,W], got {pred.shape}")
    if pred.shape[1:] == target_hw:
        return pred
    import cv2

    height, width = target_hw
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in pred])
