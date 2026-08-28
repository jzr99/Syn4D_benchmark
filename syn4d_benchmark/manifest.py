"""Manifest types and filesystem discovery for the raw Syn4D release."""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .protocol import DEFAULT_FRAME_INDICES


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    variant: str
    scene: str
    sequence: str
    camera: int
    rgb_dir: str
    depth_dir: str
    camera_csv: str
    tracking_safetensors: str
    frame_indices: list[int]
    width: int
    height: int


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def discover_sequences(root: Path, frame_indices: Iterable[int] = DEFAULT_FRAME_INDICES) -> list[SequenceRecord]:
    root = Path(root).resolve()
    selected = [int(v) for v in frame_indices]
    records: list[SequenceRecord] = []
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for scene_dir in sorted(variant_dir.iterdir()):
            png_root = scene_dir / "png"
            camera_root = scene_dir / "ground_truth" / "meta_exr_csv"
            if not png_root.is_dir() or not camera_root.is_dir():
                continue
            for rgb_dir in sorted(p for p in png_root.iterdir() if p.is_dir()):
                sequence = rgb_dir.name
                try:
                    camera = int(sequence.rsplit("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Sequence name must end in a camera integer: {sequence}") from exc
                rgb_paths = [rgb_dir / f"{sequence}_{frame:04d}.png" for frame in selected]
                missing = [str(path) for path in rgb_paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"{sequence}: missing selected RGB frame {missing[0]}")
                width, height = _png_size(rgb_paths[0])
                camera_csv = camera_root / f"{sequence}_camera.csv"
                tracking = scene_dir / "tracking_safetensors" / f"{sequence}.safetensors"
                depth_dir = scene_dir / "exr_layers" / "depth" / sequence
                if not camera_csv.is_file() or not tracking.is_file() or not depth_dir.is_dir():
                    raise FileNotFoundError(f"Incomplete benchmark inputs for {variant_dir.name}/{scene_dir.name}/{sequence}")
                records.append(
                    SequenceRecord(
                        sequence_id=f"{variant_dir.name}/{scene_dir.name}/{sequence}",
                        variant=variant_dir.name,
                        scene=scene_dir.name,
                        sequence=sequence,
                        camera=camera,
                        rgb_dir=str(rgb_dir),
                        depth_dir=str(depth_dir),
                        camera_csv=str(camera_csv),
                        tracking_safetensors=str(tracking),
                        frame_indices=selected,
                        width=int(width),
                        height=int(height),
                    )
                )
    return records


def write_manifest(records: Iterable[SequenceRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[SequenceRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [SequenceRecord(**json.loads(line)) for line in handle if line.strip()]
