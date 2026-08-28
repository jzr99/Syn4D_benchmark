#!/usr/bin/env python3
"""Run the official V-DPM model and emit canonical Syn4D predictions."""

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
    homogeneous_c2w,
    image_paths,
    load_pending,
    numpy,
    parse_tasks,
    points_world_to_camera,
    points_world_to_ref0,
    query_pixels,
    sample_nearest,
    save_prediction,
    select_records,
)
from syn4d_benchmark.data import prediction_path
from syn4d_benchmark.manifest import read_manifest

MODEL_REVISION = "5e2a57cf6007dfb0511a8b396a0805089b9edcc4"


def _load_rgb(paths: list[str]) -> list[np.ndarray]:
    import cv2

    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _build_model(repo: Path, checkpoint: Path | None, device: str):
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(repo.resolve()))
    from dpm.model import VDPM

    cfg = OmegaConf.create({"model": OmegaConf.load(repo / "configs" / "model" / "dpm.yaml")})
    model = VDPM(cfg).to(device)
    if checkpoint is None:
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt",
            file_name="vdpm_model.pt",
            progress=True,
            map_location="cpu",
        )
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _extract(result, tracking_root: Path, record, tasks: set[str]) -> dict[str, np.ndarray]:
    import torch
    from einops import repeat
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    pointmaps = result["pointmaps"]
    height, width = pointmaps[0]["pts3d"].shape[2:4]
    extrinsic, _ = pose_encoding_to_extri_intri(result["pose_enc"], (height, width))
    w2c = numpy(extrinsic[0])
    w2c = np.concatenate(
        [w2c, repeat(torch.tensor([0.0, 0.0, 0.0, 1.0]), "c -> s 1 c", s=w2c.shape[0]).numpy()],
        axis=1,
    )
    c2w = np.linalg.inv(w2c)
    arrays: dict[str, np.ndarray] = {}
    if "pose" in tasks:
        arrays["camera_c2w"] = homogeneous_c2w(c2w).astype(np.float32)
    if "tracking" in tasks:
        query = query_pixels(tracking_root, record, (height, width))
        tracks_world = np.stack(
            [sample_nearest(numpy(pointmaps[t]["pts3d"])[0, 0], query) for t in range(len(pointmaps))]
        )
        arrays["tracking_xyz"] = points_world_to_ref0(tracks_world, c2w[0]).astype(np.float32)
    if "depth" in tasks:
        depths = []
        for t, pointmap in enumerate(pointmaps):
            points_world = numpy(pointmap["pts3d"])[0, t]
            depths.append(points_world_to_camera(points_world, c2w[t])[..., 2])
        arrays["depth"] = np.stack(depths).astype(np.float32)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "v-dpm" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR / "external" / "vdpm")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="tracking,depth,pose")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks, {"tracking", "depth", "pose"})

    model = _build_model(args.repo, args.checkpoint, args.device)
    import torch
    from util.vggt import preprocess_images
    records = select_records(read_manifest(args.manifest), args)
    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays, pending = load_pending(output, tasks, args.overwrite)
        if not pending:
            print(f"[{index}/{len(records)}] resume {record.sequence_id}")
            continue
        started_at = time.time()
        images = preprocess_images(_load_rgb(image_paths(record))).to(args.device)
        with torch.no_grad():
            result = model.inference(None, images=images.unsqueeze(0))
        arrays.update(_extract(result, args.tracking_gt, record, pending))
        save_prediction(
            output,
            arrays,
            record,
            "v-dpm",
            MODEL_REVISION,
            started_at,
            {"model_hw": list(images.shape[-2:])},
        )
        print(f"[{index}/{len(records)}] {record.sequence_id} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
