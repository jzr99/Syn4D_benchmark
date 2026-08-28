from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from syn4d_benchmark.data import TRACKING_GT_SCHEMA, load_tracking_gt, save_tracking_gt
from syn4d_benchmark.manifest import SequenceRecord, write_manifest


def test_tracking_gt_structured_npy_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "sequence.npy"
    arrays = {
        "tracks_ref0": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "visibility": np.ones((2, 4), dtype=bool),
        "dynamic": np.array([False, True, False, True]),
        "query_uv": np.arange(8, dtype=np.float32).reshape(4, 2),
        "frame_indices": np.array([0, 6], dtype=np.int32),
        "intrinsics": np.array([10.0, 11.0, 5.0, 4.0]),
        "image_size": np.array([8, 10], dtype=np.int32),
    }
    save_tracking_gt(output, "variant/scene/sequence", **arrays)
    raw = np.load(output, allow_pickle=False)
    assert raw.shape == ()
    assert raw["schema"].item() == TRACKING_GT_SCHEMA
    loaded = load_tracking_gt(output)
    assert loaded["sequence_id"].item() == "variant/scene/sequence"
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)


def test_baseline_registry_points_to_runners() -> None:
    benchmark_dir = Path(__file__).resolve().parents[1]
    registry = json.loads((benchmark_dir / "baselines.json").read_text(encoding="utf-8"))
    assert set(registry) == {"v-dpm", "any4d", "traceanything", "st4rtrack", "spatrackerv2"}
    for spec in registry.values():
        assert len(spec["revision"]) == 40
        assert spec["official_repo"].startswith("https://github.com/")
        assert (benchmark_dir / spec["runner"]).is_file()


def test_tracking_evaluation_does_not_read_raw_metadata(tmp_path: Path, monkeypatch) -> None:
    from evaluate import main

    record = SequenceRecord(
        sequence_id="variant/scene/sequence_0",
        variant="variant",
        scene="scene",
        sequence="sequence_0",
        camera=0,
        rgb_dir="/intentionally/missing/rgb",
        depth_dir="/intentionally/missing/depth",
        camera_csv="/intentionally/missing/camera.csv",
        tracking_safetensors="/intentionally/missing/tracks.safetensors",
        frame_indices=[0, 6],
        width=10,
        height=8,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)

    tracks = np.arange(24, dtype=np.float32).reshape(2, 4, 3) + 1
    tracking_gt = tmp_path / "tracking_gt" / "variant" / "scene" / "sequence_0.npy"
    save_tracking_gt(
        tracking_gt,
        record.sequence_id,
        tracks_ref0=tracks,
        visibility=np.ones((2, 4), dtype=bool),
        dynamic=np.array([False, True, False, True]),
        query_uv=np.arange(8, dtype=np.float32).reshape(4, 2),
        frame_indices=np.array(record.frame_indices, dtype=np.int32),
        intrinsics=np.array([10.0, 11.0, 5.0, 4.0]),
        image_size=np.array([8, 10], dtype=np.int32),
    )
    predictions = tmp_path / "predictions" / "variant" / "scene"
    predictions.mkdir(parents=True)
    np.savez_compressed(
        predictions / "sequence_0.npz",
        tracking_xyz=tracks,
        frame_indices=np.array(record.frame_indices, dtype=np.int32),
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--manifest",
            str(manifest),
            "--predictions",
            str(tmp_path / "predictions"),
            "--tracking-gt",
            str(tmp_path / "tracking_gt"),
            "--tasks",
            "tracking",
            "--strict",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["failures"] == []
    assert summary["tasks"]["tracking"]["overall"]["score"] == 1.0
