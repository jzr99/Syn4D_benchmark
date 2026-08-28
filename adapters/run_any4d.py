#!/usr/bin/env python3
"""Run the official Any4D model and emit canonical Syn4D predictions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from adapters.common import (
    add_selection_arguments,
    image_paths,
    load_pending,
    numpy,
    parse_tasks,
    points_world_to_ref0,
    query_pixels,
    sample_nearest,
    save_prediction,
    select_records,
)
from syn4d_benchmark.data import prediction_path
from syn4d_benchmark.manifest import read_manifest

MODEL_REVISION = "aa9f1b0d7ecdf44a6ef8cef93387be0093bdd497"


def _build_model(repo: Path, checkpoint: Path, device: str):
    import hydra
    import torch

    sys.path.insert(0, str(repo.resolve()))
    from any4d.models import init_model

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str((repo / "configs").resolve())):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "machine=local",
                "model=any4d",
                "model.encoder.uses_torch_hub=false",
                "model/task=images_only",
            ],
        )
    model = init_model(cfg.model.model_str, cfg.model.model_config)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing Any4D checkpoint {checkpoint}; download any4d_4v_combined.pth from the official release"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    return model.to(device).eval()


def _camera_poses(predictions) -> np.ndarray:
    import torch
    from any4d.utils.geometry import quaternion_to_rotation_matrix

    poses = []
    for pred in predictions:
        rotation = numpy(quaternion_to_rotation_matrix(pred["cam_quats"]))[0]
        translation = numpy(pred["cam_trans"])[0]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
    return np.stack(poses)


def _extract(result, tracking_root: Path, record, tasks: set[str]) -> dict[str, np.ndarray]:
    predictions = [result[f"pred{index + 1}"] for index in range(len(record.frame_indices))]
    c2w = _camera_poses(predictions)
    arrays: dict[str, np.ndarray] = {}
    if "pose" in tasks:
        arrays["camera_c2w"] = c2w.astype(np.float32)
    if "depth" in tasks:
        arrays["depth"] = np.stack([numpy(pred["pts3d_cam"])[0, ..., 2] for pred in predictions]).astype(np.float32)
    if "tracking" in tasks:
        base = numpy(predictions[0]["pts3d"])[0]
        query = query_pixels(tracking_root, record, base.shape[:2])
        fields = [base]
        for pred in predictions[1:]:
            fields.append(base + numpy(pred["scene_flow"])[0])
        tracks_world = np.stack([sample_nearest(field, query) for field in fields])
        arrays["tracking_xyz"] = points_world_to_ref0(tracks_world, c2w[0]).astype(np.float32)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "any4d" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR / "external" / "any4d")
    parser.add_argument("--checkpoint", type=Path, default=BENCHMARK_DIR / "external" / "any4d" / "checkpoints" / "any4d_4v_combined.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="tracking,depth,pose")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks, {"tracking", "depth", "pose"})

    model = _build_model(args.repo, args.checkpoint, args.device)
    from any4d.utils.image import load_images
    from any4d.utils.inference import loss_of_one_batch_multi_view

    records = select_records(read_manifest(args.manifest), args)
    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays, pending = load_pending(output, tasks, args.overwrite)
        if not pending:
            print(f"[{index}/{len(records)}] resume {record.sequence_id}")
            continue
        started_at = time.time()
        views = load_images(
            image_paths(record),
            resize_mode="fixed_mapping",
            resolution_set=518,
            norm_type="dinov2",
            patch_size=14,
            compute_moge_mask=False,
        )
        result = loss_of_one_batch_multi_view(views, model, None, args.device, use_amp=True)
        arrays.update(_extract(result, args.tracking_gt, record, pending))
        model_hw = list(views[0]["img"].shape[-2:])
        save_prediction(output, arrays, record, "any4d", MODEL_REVISION, started_at, {"model_hw": model_hw})
        print(f"[{index}/{len(records)}] {record.sequence_id} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
