"""Shared utilities for official-model benchmark adapters."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_DIR / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--tracking-gt",
        type=Path,
        default=Path(os.environ.get("SYN4D_TRACKING_GT", BENCHMARK_DIR / "data" / "tracking_gt")),
    )
    parser.add_argument("--variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")


def parse_tasks(value: str, supported: Iterable[str]) -> set[str]:
    tasks = {item.strip() for item in value.split(",") if item.strip()}
    supported = set(supported)
    unknown = tasks - supported
    if unknown:
        raise ValueError(f"Unsupported tasks {sorted(unknown)}; this adapter supports {sorted(supported)}")
    if not tasks:
        raise ValueError("At least one task is required")
    return tasks


def select_records(records, args):
    variants = {value for value in args.variants.split(",") if value}
    scenes = {value for value in args.scenes.split(",") if value}
    cameras = {int(value) for value in args.cameras.split(",") if value}
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"shard index must be in [0, {args.num_shards}), got {args.shard_index}")
    selected = [
        record
        for record in records
        if (not variants or record.variant in variants)
        and (not scenes or record.scene in scenes)
        and (not cameras or record.camera in cameras)
    ][args.shard_index :: args.num_shards]
    return selected[: args.limit] if args.limit > 0 else selected


def video_path(record) -> Path:
    return Path(record.rgb_dir).parents[1] / "mp4" / f"{record.sequence}.mp4"


def image_paths(record) -> list[str]:
    """Return selected RGB frames, extracting them from the release MP4 if needed."""
    source_paths = [Path(record.rgb_dir) / f"{record.sequence}_{frame:04d}.png" for frame in record.frame_indices]
    if all(path.is_file() for path in source_paths):
        return [str(path) for path in source_paths]

    video = video_path(record)
    if not video.is_file():
        raise FileNotFoundError(
            f"Neither the selected PNG frames nor release video exists for {record.sequence_id}: {video}"
        )

    cache_root = Path(os.environ.get("SYN4D_FRAME_CACHE", BENCHMARK_DIR / "data" / "frame_cache"))
    output_dir = cache_root / record.variant / record.scene / record.sequence
    output_paths = [output_dir / f"{record.sequence}_{frame:04d}.png" for frame in record.frame_indices]
    if all(path.is_file() for path in output_paths):
        return [str(path) for path in output_paths]

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = dict(zip(record.frame_indices, output_paths))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open release video {video}")
    written: set[int] = set()
    frame_index = 0
    try:
        while len(written) < len(selected):
            ok, frame = capture.read()
            if not ok:
                break
            output = selected.get(frame_index)
            if output is not None:
                temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
                if not cv2.imwrite(str(temporary), frame):
                    raise RuntimeError(f"Could not write decoded frame {temporary}")
                temporary.replace(output)
                written.add(frame_index)
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(set(selected) - written)
    if missing:
        raise RuntimeError(f"Video {video} ended before benchmark frames {missing[:5]}")
    return [str(path) for path in output_paths]


def query_pixels(tracking_root: Path, record, target_hw: tuple[int, int]) -> np.ndarray:
    from syn4d_benchmark.data import load_tracking_gt, tracking_gt_path

    gt = load_tracking_gt(tracking_gt_path(tracking_root, record))
    if str(gt["sequence_id"].item()) != record.sequence_id:
        raise ValueError(f"tracking GT identity mismatch for {record.sequence_id}")
    source_h, source_w = np.asarray(gt["image_size"], dtype=np.float64)
    target_h, target_w = target_hw
    query = np.asarray(gt["query_uv"], dtype=np.float64).copy()
    query[:, 0] *= target_w / source_w
    query[:, 1] *= target_h / source_h
    return query


def sample_nearest(field: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """Sample an HxW[xC] field at floating x/y pixel coordinates."""
    field = np.asarray(field)
    if field.ndim not in (2, 3):
        raise ValueError(f"field must be [H,W] or [H,W,C], got {field.shape}")
    height, width = field.shape[:2]
    x = np.clip(np.rint(query_xy[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(query_xy[:, 1]).astype(np.int64), 0, height - 1)
    return field[y, x]


def homogeneous_c2w(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape[-2:] == (4, 4):
        return c2w
    if c2w.shape[-2:] != (3, 4):
        raise ValueError(f"camera matrices must end in [3,4] or [4,4], got {c2w.shape}")
    bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (*c2w.shape[:-2], 1, 4))
    return np.concatenate([c2w, bottom], axis=-2)


def points_world_to_ref0(points_world: np.ndarray, c2w0: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    w2c0 = np.linalg.inv(homogeneous_c2w(c2w0))
    return points_world @ w2c0[:3, :3].T + w2c0[:3, 3]


def points_world_to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    w2c = np.linalg.inv(homogeneous_c2w(c2w))
    return points_world @ w2c[:3, :3].T + w2c[:3, 3]


TASK_KEYS = {"tracking": "tracking_xyz", "depth": "depth", "pose": "camera_c2w"}


def load_pending(output: Path, tasks: set[str], overwrite: bool) -> tuple[dict[str, np.ndarray], set[str]]:
    arrays: dict[str, np.ndarray] = {}
    if output.is_file() and not overwrite:
        with np.load(output, allow_pickle=False) as existing:
            arrays = {key: np.asarray(existing[key]) for key in existing.files}
    return arrays, {task for task in tasks if TASK_KEYS[task] not in arrays}


def save_prediction(
    output: Path,
    arrays: dict[str, np.ndarray],
    record,
    model: str,
    revision: str,
    started_at: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    arrays["frame_indices"] = np.asarray(record.frame_indices, dtype=np.int32)
    arrays["model"] = np.asarray(model)
    arrays["model_revision"] = np.asarray(revision)
    arrays["runtime_seconds"] = np.asarray(
        float(np.asarray(arrays.get("runtime_seconds", 0.0))) + time.time() - started_at,
        dtype=np.float64,
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)


def numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def squeeze_batch(value, final_ndim: int) -> np.ndarray:
    array = numpy(value)
    while array.ndim > final_ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim != final_ndim:
        raise ValueError(f"Expected {final_ndim} dimensions, got {array.shape}")
    return array
