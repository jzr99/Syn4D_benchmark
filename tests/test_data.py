from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from syn4d_benchmark.data import TRACKING_GT_SCHEMA, load_tracking_gt, save_tracking_gt
from syn4d_benchmark.manifest import SequenceRecord, read_manifest, write_manifest


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


def test_read_manifest_resolves_portable_paths(tmp_path: Path) -> None:
    record = SequenceRecord(
        sequence_id="variant/scene/sequence_0",
        variant="variant",
        scene="scene",
        sequence="sequence_0",
        camera=0,
        rgb_dir="variant/scene/png/sequence_0",
        depth_dir="variant/scene/exr_layers/depth/sequence_0",
        camera_csv="variant/scene/ground_truth/meta_exr_csv/sequence_0_camera.csv",
        tracking_safetensors="variant/scene/tracking_safetensors/sequence_0.safetensors",
        frame_indices=[0, 6],
        width=10,
        height=8,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    loaded = read_manifest(manifest, tmp_path / "release")[0]
    assert loaded.rgb_dir == str((tmp_path / "release/variant/scene/png/sequence_0").resolve())
    assert loaded.camera_csv == str(
        (tmp_path / "release/variant/scene/ground_truth/meta_exr_csv/sequence_0_camera.csv").resolve()
    )


def test_image_paths_extracts_release_video(tmp_path: Path, monkeypatch) -> None:
    from adapters.common import image_paths

    rgb_dir = tmp_path / "release/variant/scene/png/sequence_0"
    video = tmp_path / "release/variant/scene/mp4/sequence_0.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    record = SequenceRecord(
        sequence_id="variant/scene/sequence_0",
        variant="variant",
        scene="scene",
        sequence="sequence_0",
        camera=0,
        rgb_dir=str(rgb_dir),
        depth_dir="unused",
        camera_csv="unused",
        tracking_safetensors="unused",
        frame_indices=[0, 2],
        width=3,
        height=2,
    )

    class Capture:
        def __init__(self, path: str):
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self.index == 3:
                return False, None
            frame = np.full((2, 3, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

        def release(self) -> None:
            pass

    def imwrite(path: str, frame: np.ndarray) -> bool:
        Path(path).write_bytes(frame.tobytes())
        return True

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Capture, imwrite=imwrite))
    monkeypatch.setenv("SYN4D_FRAME_CACHE", str(tmp_path / "cache"))
    paths = [Path(path) for path in image_paths(record)]
    assert [path.name for path in paths] == ["sequence_0_0000.png", "sequence_0_0002.png"]
    assert all(path.is_file() for path in paths)
