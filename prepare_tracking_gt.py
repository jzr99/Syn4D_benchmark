#!/usr/bin/env python3
"""Build fixed sparse tracking GT from the existing Syn4D→WorldTrack converter."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from syn4d_benchmark.data import save_tracking_gt
from syn4d_benchmark.manifest import SequenceRecord, read_manifest
from syn4d_benchmark.protocol import TRACKING_DYNAMIC_THRESHOLD_METERS


def _stable_rng(sequence_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _locate_dense_npz(dense_subset: Path, record: SequenceRecord) -> Path:
    suffix = f"{record.scene}{record.sequence}.npz"
    matches = [path for path in dense_subset.glob("*.npz") if path.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dense NPZ ending {suffix} in {dense_subset}, found {len(matches)}")
    return matches[0]


def _recover_source_intrinsics(pack, record: SequenceRecord) -> np.ndarray:
    """Undo the converter's centered square crop and resize."""
    import cv2

    encoded = np.frombuffer(pack["images_jpeg_bytes"][0], dtype=np.uint8)
    dense_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if dense_image is None:
        raise RuntimeError(f"Could not decode dense image for {record.sequence_id}")
    dense_side = min(dense_image.shape[:2])
    source_side = min(record.height, record.width)
    scale = source_side / dense_side
    dense_intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
    return np.array(
        [
            dense_intrinsics[0] * scale,
            dense_intrinsics[1] * scale,
            record.width / 2.0,
            record.height / 2.0,
        ],
        dtype=np.float64,
    )


def _make_sparse_gt(dense_path: Path, record: SequenceRecord, queries: int, seed: int, output: Path) -> None:
    pack = np.load(dense_path, allow_pickle=True)
    xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    w2c = np.asarray(pack["extrinsics_w2c"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    ref0_from_cam = w2c[0][None] @ c2w
    tracks_ref0 = np.einsum("tij,tqj->tqi", ref0_from_cam[:, :3, :3], xyz_cam) + ref0_from_cam[:, None, :3, 3]
    tracks_world = np.einsum("tij,tqj->tqi", c2w[:, :3, :3], xyz_cam) + c2w[:, None, :3, 3]
    valid = visibility & np.isfinite(tracks_ref0).all(-1)
    step = np.linalg.norm(np.diff(tracks_world, axis=0), axis=-1)
    step_valid = valid[:-1] & valid[1:]
    dynamic = np.where(step_valid, step, 0.0).sum(0) > TRACKING_DYNAMIC_THRESHOLD_METERS

    candidates = np.flatnonzero(valid[0] & np.isfinite(xyz_cam[0, :, 2]) & (xyz_cam[0, :, 2] > 1e-8))
    if candidates.size > queries:
        candidates = np.sort(_stable_rng(record.sequence_id, seed).choice(candidates, queries, replace=False))
    intr = _recover_source_intrinsics(pack, record)
    q0 = xyz_cam[0, candidates]
    query_uv = np.stack(
        [intr[0] * q0[:, 0] / q0[:, 2] + intr[2], intr[1] * q0[:, 1] / q0[:, 2] + intr[3]],
        axis=-1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tracking_gt(
        output,
        record.sequence_id,
        tracks_ref0=tracks_ref0[:, candidates].astype(np.float32),
        visibility=valid[:, candidates],
        dynamic=dynamic[candidates],
        query_uv=query_uv.astype(np.float32),
        frame_indices=np.asarray(record.frame_indices, dtype=np.int32),
        intrinsics=intr,
        image_size=np.asarray([record.height, record.width], dtype=np.int32),
    )


def main() -> int:
    benchmark_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=benchmark_dir / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--dense-root", type=Path, default=benchmark_dir / "data" / "worldtrack_dense")
    parser.add_argument("--dense-subset-template", default="syn4d_all_{variant}")
    parser.add_argument("--output", type=Path, default=benchmark_dir / "data" / "tracking_gt")
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument(
        "--converter",
        type=Path,
        help=(
            "path to syn4d_to_worldtrack.py; required for dense conversion when "
            "the benchmark is not inside the source kaggle repository"
        ),
    )
    parser.add_argument("--variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metadata-root", default="/scratch/shared/beegfs/zeren/Syn4D/metadata")
    parser.add_argument("--fallback-metadata-root", default="/scratch/shared/beegfs/kelvin/Syn4D/metadata")
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    variants = {value for value in args.variants.split(",") if value}
    scenes = {value for value in args.scenes.split(",") if value}
    cameras = {int(value) for value in args.cameras.split(",") if value}
    records = [
        record for record in records
        if (not variants or record.variant in variants)
        and (not scenes or record.scene in scenes)
        and (not cameras or record.camera in cameras)
    ]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No manifest records selected")
    grouped: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.variant].append(record)

    raw_root = Path(records[0].rgb_dir).parents[3]
    for variant, variant_records in sorted(grouped.items()):
        subset = args.dense_subset_template.format(variant=variant)
        if not args.skip_convert:
            converter = args.converter or (
                benchmark_dir.parent / "scripts" / "eval_open_d4rt" / "syn4d_to_worldtrack.py"
            )
            if not converter.is_file():
                raise FileNotFoundError(
                    f"Syn4D converter not found at {converter}. Pass --converter PATH, "
                    "or use --skip-convert with existing dense packs."
                )
            scenes = sorted({record.scene for record in variant_records})
            command = [
                sys.executable,
                str(converter),
                "--dataset-root", str(raw_root / variant),
                "--metadata-root", args.metadata_root,
                "--fallback-metadata-root", args.fallback_metadata_root,
                "--scene-name", ",".join(scenes),
                "--num-sequences", str(len(variant_records)),
                "--num-frames", str(len(variant_records[0].frame_indices)),
                "--stride", str(variant_records[0].frame_indices[1] - variant_records[0].frame_indices[0]),
                "--resolution", "256",
                "--max-queries", "4096",
                "--output-dir", str(args.dense_root),
                "--subset-name", subset,
                "--tracking-format", "safetensor",
                "--rgb-source", "png",
                "--seed", str(args.seed),
            ]
            subprocess.run(command, check=True)
        dense_subset = args.dense_root / subset
        for index, record in enumerate(variant_records, 1):
            output = args.output / record.variant / record.scene / f"{record.sequence}.npy"
            if output.is_file():
                continue
            dense_path = _locate_dense_npz(dense_subset, record)
            _make_sparse_gt(dense_path, record, args.queries, args.seed, output)
            print(f"[{variant} {index}/{len(variant_records)}] {record.sequence_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
