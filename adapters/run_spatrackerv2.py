#!/usr/bin/env python3
"""Run official SpaTrackerV2 and emit tracking, depth, and pose predictions."""

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
    points_world_to_ref0,
    query_pixels,
    save_prediction,
    select_records,
)
from syn4d_benchmark.data import prediction_path
from syn4d_benchmark.manifest import read_manifest

MODEL_REVISION = "7e12274c52077860cebfe007a6290777db43b63c"


def _load_video(paths: list[str]):
    import cv2
    import torch

    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "spatrackerv2" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR / "external" / "spatrackerv2")
    parser.add_argument("--front-checkpoint", default="Yuxihenry/SpatialTrackerV2_Front")
    parser.add_argument("--tracker-checkpoint", default="Yuxihenry/SpatialTrackerV2-Offline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vo-points", type=int, default=756)
    parser.add_argument("--track-iterations", type=int, default=4)
    parser.add_argument("--tasks", default="tracking,depth,pose")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks, {"tracking", "depth", "pose"})

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    front = VGGT4Track.from_pretrained(args.front_checkpoint).to(args.device).eval()
    tracker = Predictor.from_pretrained(args.tracker_checkpoint)
    tracker.spatrack.track_num = args.vo_points
    tracker.eval()
    tracker.to(args.device)

    records = select_records(read_manifest(args.manifest), args)
    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays, pending = load_pending(output, tasks, args.overwrite)
        if not pending:
            print(f"[{index}/{len(records)}] resume {record.sequence_id}")
            continue
        started_at = time.time()
        video = preprocess_image(_load_video(image_paths(record)))[None]
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            front_prediction = front(video.to(args.device) / 255.0)
        c2w_seed = numpy(front_prediction["poses_pred"].squeeze(0))
        intrinsics = numpy(front_prediction["intrs"].squeeze(0))
        point_seed = numpy(front_prediction["points_map"].squeeze(0))
        depth_seed = point_seed[..., 2]
        uncertainty = numpy(front_prediction["unc_metric"].squeeze(0)) > 0.5
        model_h, model_w = video.shape[-2:]
        query = query_pixels(args.tracking_gt, record, (model_h, model_w))
        query_xyt = np.concatenate([np.zeros((len(query), 1)), query], axis=1).astype(np.float32)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                c2w,
                _,
                point_map,
                _,
                tracks_camera,
                _,
                _,
                _,
                _,
            ) = tracker.forward(
                video.squeeze(0),
                depth=depth_seed,
                intrs=intrinsics,
                extrs=c2w_seed,
                queries=query_xyt,
                fps=1,
                # Preserve the benchmark's exact fixed queries. The official
                # full_point=False path replaces low-confidence queries with
                # random valid ones, which would invalidate query identities.
                full_point=True,
                iters_track=args.track_iterations,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                unc_metric=uncertainty,
                support_frame=len(record.frame_indices) - 1,
                replace_ratio=0.2,
            )
        c2w_np = homogeneous_c2w(numpy(c2w))
        if "pose" in pending:
            arrays["camera_c2w"] = c2w_np.astype(np.float32)
        if "depth" in pending:
            points = numpy(point_map)
            if points.ndim != 4 or points.shape[1] != 3:
                raise ValueError(f"Unexpected SpaTrackerV2 point map shape {points.shape}")
            arrays["depth"] = points[:, 2].astype(np.float32)
        if "tracking" in pending:
            tracks_camera_np = numpy(tracks_camera)[..., :3]
            tracks_world = np.einsum("tij,tqj->tqi", c2w_np[:, :3, :3], tracks_camera_np) + c2w_np[:, None, :3, 3]
            arrays["tracking_xyz"] = points_world_to_ref0(tracks_world, c2w_np[0]).astype(np.float32)
        save_prediction(
            output,
            arrays,
            record,
            "spatrackerv2",
            MODEL_REVISION,
            started_at,
            {
                "model_hw": [model_h, model_w],
                "front_checkpoint": args.front_checkpoint,
                "tracker_checkpoint": args.tracker_checkpoint,
            },
        )
        print(f"[{index}/{len(records)}] {record.sequence_id} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
