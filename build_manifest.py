#!/usr/bin/env python3
"""Index every complete sequence under the raw Syn4D kaggle_eval root."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from syn4d_benchmark.manifest import discover_sequences, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/work/kelvin/Syn4D/subsets/kaggle_eval"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "manifests" / "syn4d_all.jsonl")
    parser.add_argument("--frames", default="0:192:6", help="Python-style start:stop:stride")
    args = parser.parse_args()
    start, stop, stride = (int(v) for v in args.frames.split(":"))
    records = discover_sequences(args.root, range(start, stop, stride))
    if not records:
        raise RuntimeError(f"No complete sequences found below {args.root}")
    write_manifest(records, args.output)
    counts = Counter((r.variant, r.scene) for r in records)
    print(f"wrote {len(records)} sequences to {args.output}")
    for (variant, scene), count in sorted(counts.items()):
        print(f"  {variant}/{scene}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
