#!/usr/bin/env python3
"""Run official St4RTrack sequence inference on the Syn4D tracking task."""

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
    query_pixels,
    sample_nearest,
    save_prediction,
    select_records,
)
from syn4d_benchmark.data import prediction_path
from syn4d_benchmark.manifest import read_manifest

MODEL_REVISION = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"


def _gather_anchor_pointmaps(output) -> np.ndarray:
    chunks = []
    for result in output:
        points = numpy(result["pred1"]["pts3d"])
        while points.ndim > 4 and points.shape[0] == 1:
            points = points[0]
        if points.ndim == 3:
            points = points[None]
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError(f"Unexpected St4RTrack pointmap shape {points.shape}")
        chunks.append(points)
    return np.concatenate(chunks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "st4rtrack" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR / "external" / "st4rtrack")
    parser.add_argument("--checkpoint", default="yupengchengg147/St4RTrack")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tasks", default="tracking")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks, {"tracking"})

    sys.path.insert(0, str(args.repo.resolve()))
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    model = AsymmetricCroCo3DStereo.from_pretrained(args.checkpoint).to(args.device).eval()
    records = select_records(read_manifest(args.manifest), args)
    for index, record in enumerate(records, 1):
        output_path = prediction_path(args.output, record)
        arrays, pending = load_pending(output_path, tasks, args.overwrite)
        if not pending:
            print(f"[{index}/{len(records)}] resume {record.sequence_id}")
            continue
        started_at = time.time()
        views = load_images(
            image_paths(record),
            size=args.image_size,
            square_ok=True,
            crop=False,
            num_frames=len(record.frame_indices),
            verbose=False,
        )
        output = inference(views, model, args.device, batch_size=args.batch_size, anchor_view=0)
        fields = _gather_anchor_pointmaps(output)
        if len(fields) != len(record.frame_indices):
            raise ValueError(f"St4RTrack returned {len(fields)} fields for {len(record.frame_indices)} frames")
        query = query_pixels(args.tracking_gt, record, fields.shape[1:3])
        arrays["tracking_xyz"] = np.stack([sample_nearest(field, query) for field in fields]).astype(np.float32)
        save_prediction(
            output_path,
            arrays,
            record,
            "st4rtrack",
            MODEL_REVISION,
            started_at,
            {"model_hw": list(fields.shape[1:3]), "checkpoint": args.checkpoint},
        )
        print(f"[{index}/{len(records)}] {record.sequence_id} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
