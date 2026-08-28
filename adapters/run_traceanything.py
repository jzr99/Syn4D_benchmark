#!/usr/bin/env python3
"""Run official TraceAnything trajectory fields on the Syn4D tracking task."""

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

MODEL_REVISION = "54677b5e7bf11510c2e8c917a509988ad379f8eb"


def _build_model(repo: Path, config: Path, checkpoint: Path, device):
    sys.path.insert(0, str(repo.resolve()))
    from scripts.infer import _build_model_from_cfg, _load_cfg

    return _build_model_from_cfg(_load_cfg(str(config)), str(checkpoint), device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "traceanything" / "predictions")
    parser.add_argument("--repo", type=Path, default=BENCHMARK_DIR / "external" / "traceanything")
    parser.add_argument("--config", type=Path, default=BENCHMARK_DIR / "external" / "traceanything" / "configs" / "eval.yaml")
    parser.add_argument("--checkpoint", type=Path, default=BENCHMARK_DIR / "external" / "traceanything" / "checkpoints" / "trace_anything.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="tracking")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks, {"tracking"})

    import torch

    device = torch.device(args.device)
    model = _build_model(args.repo, args.config, args.checkpoint, device)
    from scripts.infer import _load_images

    records = select_records(read_manifest(args.manifest), args)
    for index, record in enumerate(records, 1):
        output = prediction_path(args.output, record)
        arrays, pending = load_pending(output, tasks, args.overwrite)
        if not pending:
            print(f"[{index}/{len(records)}] resume {record.sequence_id}")
            continue
        started_at = time.time()
        views = _load_images_from_paths(_load_images, image_paths(record), device)
        with torch.no_grad():
            predictions = model.forward(views)
        source = predictions[0]
        # The official forward pass evaluates the B-spline at the model's
        # learned time for every target frame. Preserve those learned times.
        fields = np.concatenate([numpy(field) for field in source["track_pts3d"]], axis=0)
        if fields.shape[0] != len(record.frame_indices) or fields.shape[-1] != 3:
            raise ValueError(f"Unexpected TraceAnything trajectory field shape {fields.shape}")
        query = query_pixels(args.tracking_gt, record, fields.shape[1:3])
        arrays["tracking_xyz"] = np.stack([sample_nearest(field, query) for field in fields]).astype(np.float32)
        save_prediction(
            output,
            arrays,
            record,
            "traceanything",
            MODEL_REVISION,
            started_at,
            {"model_hw": list(fields.shape[1:3]), "source_view": 0},
        )
        print(f"[{index}/{len(records)}] {record.sequence_id} -> {output}")
    return 0


def _load_images_from_paths(loader, paths: list[str], device):
    """Use the official loader without materializing a temporary scene directory."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="syn4d-traceanything-") as directory:
        for index, source in enumerate(paths):
            os.symlink(source, Path(directory) / f"{index:03d}.png")
        return loader(directory, device)


if __name__ == "__main__":
    raise SystemExit(main())
