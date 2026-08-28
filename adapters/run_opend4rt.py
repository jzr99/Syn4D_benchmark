#!/usr/bin/env python3
"""Run Open-D4RT and write the benchmark's canonical prediction NPZs."""

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


def _load_video(record: SequenceRecord) -> np.ndarray:
    import cv2

    frames = []
    for frame in record.frame_indices:
        path = Path(record.rgb_dir) / f"{record.sequence}_{frame:04d}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def _predict_depth(model, video_model: np.ndarray, grid_size: int, chunk_size: int) -> np.ndarray:
    import torch

    from infer_track_3d import _encode_model_memory
    from src.eval.tasks import _run_model_for_queries

    device = next(model.parameters()).device
    num_frames = video_model.shape[0]
    coords = np.linspace(0.0, 1.0, num=int(grid_size), dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, indexing="xy"), axis=-1).reshape(-1, 2)
    points = grid.shape[0]
    uv = np.tile(grid, (num_frames, 1))
    frame_ids = np.repeat(np.arange(num_frames, dtype=np.int64), points)
    video_tensor = torch.from_numpy(video_model).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)[None] / 255.0
    aspect = torch.tensor([[video_model.shape[2] / video_model.shape[1]]], device=device, dtype=torch.float32)
    memory = _encode_model_memory(model=model, video_b=video_tensor, aspect_b=aspect)
    query = {
        "u": torch.from_numpy(uv[:, 0]).to(device),
        "v": torch.from_numpy(uv[:, 1]).to(device),
        "t_src": torch.from_numpy(frame_ids).to(device),
        "t_tgt": torch.from_numpy(frame_ids).to(device),
        "t_cam": torch.from_numpy(frame_ids).to(device),
    }
    pred = _run_model_for_queries(model, video_tensor, aspect, query, chunk_size, memory)
    return pred["xyz_3d"].numpy()[:, 2].reshape(num_frames, grid_size, grid_size).astype(np.float32)


def _select(
    records: list[SequenceRecord],
    variants: str,
    scenes: str,
    cameras: str,
    limit: int,
    shard_index: int,
    num_shards: int,
) -> list[SequenceRecord]:
    variant_set = {v for v in variants.split(",") if v}
    scene_set = {v for v in scenes.split(",") if v}
    camera_set = {int(v) for v in cameras.split(",") if v}
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(f"shard index must be in [0, {num_shards}), got {shard_index}")
    selected = [r for r in records if (not variant_set or r.variant in variant_set) and (not scene_set or r.scene in scene_set) and (not camera_set or r.camera in camera_set)]
    selected = selected[shard_index::num_shards]
    return selected[:limit] if limit > 0 else selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=BENCHMARK_DIR / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--tracking-gt", type=Path, default=BENCHMARK_DIR / "data" / "tracking_gt")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "opend4rt" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR.parent / "Open-d4rt")
    parser.add_argument("--checkpoint", type=Path, default=BENCHMARK_DIR.parent / "Open-d4rt" / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "opend4rt.ckpt")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--tasks", default="tracking,depth,pose")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--depth-grid-size", type=int, default=128)
    parser.add_argument("--camera-grid-size", type=int, default=16)
    parser.add_argument("--variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tasks = {v.strip() for v in args.tasks.split(",") if v.strip()}
    if tasks - {"tracking", "depth", "pose"}:
        raise ValueError(f"Unknown tasks: {tasks}")

    sys.path.insert(0, str(args.repo.resolve()))
    import torch

    from infer_track_3d import _infer_tracks, _resize_video, _resolve_device, _unwrap_state_dict
    from src.core import load_checkpoint, load_yaml_config
    from src.model import build_model
    from vis.build_like_demo import _predict_camera_branches

    config_path = args.model_config or args.checkpoint.parent / "model.yaml"
    cfg = load_yaml_config(config_path)
    model = build_model(cfg["model"]).eval()
    model.load_state_dict(_unwrap_state_dict(load_checkpoint(args.checkpoint, map_location="cpu")), strict=False)
    device = _resolve_device(args.device)
    model.to(device).eval()
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    model_hw = (int(image_size[0]), int(image_size[1]))
    records = _select(
        read_manifest(args.manifest),
        args.variants,
        args.scenes,
        args.cameras,
        args.limit,
        args.shard_index,
        args.num_shards,
    )
    task_keys = {"tracking": "tracking_xyz", "depth": "depth", "pose": "camera_c2w"}

    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays: dict[str, np.ndarray] = {}
        pending_tasks = set(tasks)
        if output.is_file() and not args.overwrite:
            with np.load(output, allow_pickle=False) as existing:
                arrays = {key: np.asarray(existing[key]) for key in existing.files}
            pending_tasks = {task for task in tasks if task_keys[task] not in arrays}
            if not pending_tasks:
                print(f"[{index}/{len(records)}] resume {record.sequence_id}")
                continue
        start = time.time()
        video = _load_video(record)
        video_model = _resize_video(video, image_hw=model_hw)
        arrays.update({"frame_indices": np.asarray(record.frame_indices, dtype=np.int32), "model": np.asarray("opend4rt")})
        with torch.no_grad():
            if "tracking" in pending_tasks:
                gt = load_tracking_gt(tracking_gt_path(args.tracking_gt, record))
                query_uv = np.asarray(gt["query_uv"], dtype=np.float32)
                query_uv[:, 0] /= max(record.width - 1, 1)
                query_uv[:, 1] /= max(record.height - 1, 1)
                payload = _infer_tracks(model=model, video_model_rgb=video_model, query_uv_norm=np.clip(query_uv, 0, 1), query_chunk_size=args.query_chunk_size)
                arrays["tracking_xyz"] = np.asarray(payload["tracks_xyz_ref0"], dtype=np.float32).transpose(1, 0, 2)
            if "depth" in pending_tasks:
                arrays["depth"] = _predict_depth(model, video_model, args.depth_grid_size, args.query_chunk_size)
            if "pose" in pending_tasks:
                camera = _predict_camera_branches(
                    model=model,
                    video_model_rgb=video_model,
                    image_hw=model_hw,
                    camera_grid_size=args.camera_grid_size,
                    camera_query_chunk_size=args.query_chunk_size,
                    predict_intrinsics=False,
                    predict_extrinsics=True,
                )
                if camera is None or not np.all(camera["valid_extrinsics"]):
                    raise RuntimeError(f"Open-D4RT did not produce all camera poses for {record.sequence_id}")
                arrays["camera_c2w"] = np.asarray(camera["T_ref0_cam"], dtype=np.float32)
        previous_runtime = float(np.asarray(arrays.get("runtime_seconds", 0.0)))
        arrays["runtime_seconds"] = np.asarray(previous_runtime + time.time() - start, dtype=np.float64)
        arrays["metadata_json"] = np.asarray(json.dumps({"model_hw": model_hw, "depth_grid_size": args.depth_grid_size}))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        print(f"[{index}/{len(records)}] {record.sequence_id} {float(arrays['runtime_seconds']):.1f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
