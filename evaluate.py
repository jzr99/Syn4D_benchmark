#!/usr/bin/env python3
"""Evaluate canonical per-sequence prediction NPZs on one or more tasks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from syn4d_benchmark.data import (
    load_camera_gt,
    load_depth_gt,
    load_tracking_gt,
    prediction_path,
    resize_depth_video,
    tracking_gt_path,
)
from syn4d_benchmark.manifest import SequenceRecord, read_manifest
from syn4d_benchmark.metrics import depth_metrics, pose_metrics, tracking_metrics


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    result = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = float(np.mean(values))
    result["num_sequences"] = len(rows)
    return result


def _filter(records: list[SequenceRecord], variants: str, scenes: str, cameras: str) -> list[SequenceRecord]:
    variant_set = {v for v in variants.split(",") if v}
    scene_set = {v for v in scenes.split(",") if v}
    camera_set = {int(v) for v in cameras.split(",") if v}
    return [
        r for r in records
        if (not variant_set or r.variant in variant_set)
        and (not scene_set or r.scene in scene_set)
        and (not camera_set or r.camera in camera_set)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracking-gt", type=Path, default=Path(__file__).parent / "data" / "tracking_gt")
    parser.add_argument("--tasks", default="tracking,depth,pose")
    parser.add_argument("--depth-align", choices=("scale", "scale_shift", "metric"), default="scale")
    parser.add_argument("--variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Fail instead of recording missing predictions/GT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    unknown = tasks - {"tracking", "depth", "pose"}
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    records = _filter(read_manifest(args.manifest), args.variants, args.scenes, args.cameras)
    if args.limit > 0:
        records = records[: args.limit]
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, str]] = []

    for index, record in enumerate(records, 1):
        pred_path = prediction_path(args.predictions, record)
        if not pred_path.is_file():
            failure = {"sequence_id": record.sequence_id, "reason": f"missing prediction {pred_path}"}
            if args.strict:
                raise FileNotFoundError(failure["reason"])
            failures.append(failure)
            continue
        pred = np.load(pred_path, allow_pickle=False)
        manifest_frames = np.asarray(record.frame_indices, dtype=np.int32)
        for task in sorted(tasks):
            try:
                if task == "tracking":
                    gt_path = tracking_gt_path(args.tracking_gt, record)
                    gt = load_tracking_gt(gt_path)
                    if str(gt["sequence_id"].item()) != record.sequence_id:
                        raise ValueError("tracking GT sequence_id does not match the manifest record")
                    if "frame_indices" in pred.files and not np.array_equal(
                        np.asarray(pred["frame_indices"]), gt["frame_indices"]
                    ):
                        raise ValueError("prediction frame_indices do not match the tracking GT")
                    metrics = tracking_metrics(gt["tracks_ref0"], pred["tracking_xyz"], gt["visibility"], gt["dynamic"])
                elif task == "depth":
                    if "frame_indices" in pred.files and not np.array_equal(
                        np.asarray(pred["frame_indices"]), manifest_frames
                    ):
                        raise ValueError("prediction frame_indices do not match the manifest")
                    gt_depth = load_depth_gt(record)
                    pred_depth = resize_depth_video(pred["depth"], gt_depth.shape[1:])
                    metrics = depth_metrics(gt_depth, pred_depth, align=args.depth_align)
                else:
                    if "frame_indices" in pred.files and not np.array_equal(
                        np.asarray(pred["frame_indices"]), manifest_frames
                    ):
                        raise ValueError("prediction frame_indices do not match the manifest")
                    metrics = pose_metrics(load_camera_gt(record), pred["camera_c2w"])
                task_rows[task].append({"sequence_id": record.sequence_id, "variant": record.variant, "scene": record.scene, **metrics})
            except Exception as exc:
                failure = {"sequence_id": record.sequence_id, "task": task, "reason": f"{type(exc).__name__}: {exc}"}
                if args.strict:
                    raise RuntimeError(f"{record.sequence_id} {task} failed") from exc
                failures.append(failure)
        pred.close()
        print(f"[{index}/{len(records)}] {record.sequence_id}")

    summary: dict[str, Any] = {"protocol": "syn4d-multitask-v1", "requested_sequences": len(records), "tasks": {}}
    for task, rows in task_rows.items():
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_variant[str(row["variant"])].append(row)
            by_scene[str(row["scene"])].append(row)
        summary["tasks"][task] = {
            "overall": _mean_metrics(rows),
            "by_variant": {key: _mean_metrics(value) for key, value in sorted(by_variant.items())},
            "by_scene": {key: _mean_metrics(value) for key, value in sorted(by_scene.items())},
            "sequences": rows,
        }
    summary["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({task: value["overall"] for task, value in summary["tasks"].items()}, indent=2))
    print(f"wrote {args.output}; failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
