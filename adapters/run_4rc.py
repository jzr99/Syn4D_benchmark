#!/usr/bin/env python3
"""Run the official 4RC model and write canonical benchmark prediction NPZs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from syn4d_benchmark.data import load_tracking_gt, prediction_path, tracking_gt_path
from syn4d_benchmark.manifest import SequenceRecord, read_manifest


def _selected(records, variants: str, scenes: str, cameras: str, shard_index: int, num_shards: int, limit: int):
    variant_set = {value for value in variants.split(",") if value}
    scene_set = {value for value in scenes.split(",") if value}
    camera_set = {int(value) for value in cameras.split(",") if value}
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(f"shard index must be in [0, {num_shards}), got {shard_index}")
    result = [
        record for record in records
        if (not variant_set or record.variant in variant_set)
        and (not scene_set or record.scene in scene_set)
        and (not camera_set or record.camera in camera_set)
    ][shard_index::num_shards]
    return result[:limit] if limit > 0 else result


def _numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _squeeze_batch(value, final_ndim: int) -> np.ndarray:
    array = _numpy(value)
    while array.ndim > final_ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim != final_ndim:
        raise ValueError(f"Unexpected 4RC output shape {array.shape}; expected {final_ndim} dimensions")
    return array


def _infer(model, image_paths: list[str], size: int, device):
    import torch
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images

    views = load_images(image_paths, size=int(size), patch_size=14, verbose=False)
    for view in views:
        view["track_query_idx"] = torch.tensor([0])
    with torch.no_grad():
        output, profiling = inference(
            views,
            model,
            device,
            dtype="bf16-mixed",
            profiling=True,
            verbose=False,
            use_center_as_anchor=False,
        )
    return output["preds"], profiling


def _extract(predictions, record: SequenceRecord, tracking_gt: Path, tasks: set[str]) -> dict[str, np.ndarray]:
    c2w = np.stack([_squeeze_batch(pred["extrinsic"], 2) for pred in predictions]).astype(np.float64)
    arrays: dict[str, np.ndarray] = {}
    if "pose" in tasks:
        arrays["camera_c2w"] = c2w.astype(np.float32)
    if "depth" in tasks:
        depth = []
        for frame, pred in enumerate(predictions):
            points_world = _squeeze_batch(pred["pts"], 3).astype(np.float64)
            w2c = np.linalg.inv(c2w[frame])
            points_camera = points_world @ w2c[:3, :3].T + w2c[:3, 3]
            depth.append(points_camera[..., 2])
        arrays["depth"] = np.stack(depth).astype(np.float32)
    if "tracking" in tasks:
        gt = load_tracking_gt(tracking_gt_path(tracking_gt, record))
        query_uv = np.asarray(gt["query_uv"], dtype=np.float64)
        dense_tracks_world = np.stack([_squeeze_batch(pred["track"], 3) for pred in predictions]).astype(np.float64)
        pred_h, pred_w = dense_tracks_world.shape[1:3]
        x = np.clip((query_uv[:, 0] * pred_w / record.width).astype(np.int64), 0, pred_w - 1)
        y = np.clip((query_uv[:, 1] * pred_h / record.height).astype(np.int64), 0, pred_h - 1)
        tracks_world = dense_tracks_world[:, y, x]
        w2c0 = np.linalg.inv(c2w[0])
        arrays["tracking_xyz"] = (tracks_world @ w2c0[:3, :3].T + w2c0[:3, 3]).astype(np.float32)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_DIR / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--tracking-gt", type=Path, default=BENCHMARK_DIR / "data" / "tracking_gt")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "4rc" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR.parent / "4RC")
    parser.add_argument("--checkpoint", default="Luo-Yihang/4RC")
    parser.add_argument("--tasks", default="tracking,depth,pose")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tasks = {value.strip() for value in args.tasks.split(",") if value.strip()}
    if tasks - {"tracking", "depth", "pose"}:
        raise ValueError(f"Unknown tasks: {tasks}")
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from arc.models.arc import Arc

    device = torch.device(args.device)
    model = Arc.from_pretrained(args.checkpoint).to(device).eval()
    records = _selected(
        read_manifest(args.manifest), args.variants, args.scenes, args.cameras,
        args.shard_index, args.num_shards, args.limit,
    )
    task_keys = {"tracking": "tracking_xyz", "depth": "depth", "pose": "camera_c2w"}
    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays: dict[str, np.ndarray] = {}
        pending = set(tasks)
        if output.is_file() and not args.overwrite:
            with np.load(output, allow_pickle=False) as existing:
                arrays = {key: np.asarray(existing[key]) for key in existing.files}
            pending = {task for task in tasks if task_keys[task] not in arrays}
            if not pending:
                print(f"[{index}/{len(records)}] resume {record.sequence_id}")
                continue
        image_paths = [str(Path(record.rgb_dir) / f"{record.sequence}_{frame:04d}.png") for frame in record.frame_indices]
        start = time.time()
        try:
            predictions, profiling = _infer(model, image_paths, args.size, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            fallback = min(args.size, 336)
            print(f"OOM at size={args.size}; retrying {record.sequence_id} at size={fallback}")
            predictions, profiling = _infer(model, image_paths, fallback, device)
        arrays.update(_extract(predictions, record, args.tracking_gt, pending))
        arrays["frame_indices"] = np.asarray(record.frame_indices, dtype=np.int32)
        arrays["model"] = np.asarray("4rc")
        arrays["model_revision"] = np.asarray("a08ca0a5394c7c5f6a2a3423223cb98f85f84e6a")
        previous_runtime = float(np.asarray(arrays.get("runtime_seconds", 0.0)))
        arrays["runtime_seconds"] = np.asarray(previous_runtime + time.time() - start, dtype=np.float64)
        arrays["metadata_json"] = np.asarray(json.dumps({"size": args.size, "profiling": profiling}))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        print(f"[{index}/{len(records)}] {record.sequence_id} {float(arrays['runtime_seconds']):.1f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
