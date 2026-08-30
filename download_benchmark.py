#!/usr/bin/env python3
"""Download and validate the public Syn4D evaluation release."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "Syn4D/Syn4D_Benchmark"
EXPECTED_SEQUENCES = 512
EXPECTED_FRAMES = 32


def _tasks(value: str) -> set[str]:
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected - {"tracking", "depth", "pose"}
    if unknown or not selected:
        raise argparse.ArgumentTypeError(f"tasks must use tracking,depth,pose; got {sorted(selected)}")
    return selected


def _patterns(tasks: set[str]) -> list[str]:
    patterns = ["benchmark/**", "challenge_eval/*/*/mp4/*.mp4"]
    if "depth" in tasks:
        patterns.append("challenge_eval/*/*/exr_layers/depth/**/*.exr")
    if "pose" in tasks:
        patterns.append("challenge_eval/*/*/ground_truth/meta_exr_csv/*.csv")
    return patterns


def validate_release(root: Path, tasks: set[str]) -> dict[str, int]:
    data_root = root / "challenge_eval"
    counts = {
        "videos": len(list(data_root.glob("*/*/mp4/*.mp4"))),
        "tracking": len(list((root / "benchmark" / "data" / "tracking_gt").glob("*/*/*.npy"))),
        "depth": len(list(data_root.glob("*/*/exr_layers/depth/*/*.exr"))),
        "pose": len(list(data_root.glob("*/*/ground_truth/meta_exr_csv/*.csv"))),
    }
    expected = {"videos": EXPECTED_SEQUENCES}
    if "tracking" in tasks:
        expected["tracking"] = EXPECTED_SEQUENCES
    if "depth" in tasks:
        expected["depth"] = EXPECTED_SEQUENCES * EXPECTED_FRAMES
    if "pose" in tasks:
        expected["pose"] = EXPECTED_SEQUENCES
    failures = {name: (counts[name], count) for name, count in expected.items() if counts[name] != count}
    if failures:
        details = ", ".join(f"{name}={actual} (expected {wanted})" for name, (actual, wanted) in failures.items())
        raise RuntimeError(f"Incomplete Syn4D benchmark download: {details}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "release")
    parser.add_argument("--tasks", type=_tasks, default=_tasks("tracking,depth,pose"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=_patterns(args.tasks),
            local_dir=output,
        )
    counts = validate_release(output, args.tasks)
    print(f"Syn4D release validated at {output}")
    print(" ".join(f"{name}={count}" for name, count in counts.items()))
    print(f'export SYN4D_DATA_ROOT="{output / "challenge_eval"}"')
    print(f'export SYN4D_TRACKING_GT="{output / "benchmark" / "data" / "tracking_gt"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
