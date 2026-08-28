#!/usr/bin/env python3
"""Clone the official baseline repositories at benchmarked revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="baseline keys; default: all")
    parser.add_argument("--external-root", type=Path, default=root / "external")
    args = parser.parse_args()
    registry = json.loads((root / "baselines.json").read_text(encoding="utf-8"))
    selected = args.models or list(registry)
    unknown = set(selected) - registry.keys()
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    args.external_root.mkdir(parents=True, exist_ok=True)
    for key in selected:
        spec = registry[key]
        destination = args.external_root / key.replace("-", "")
        if not (destination / ".git").is_dir():
            run("git", "clone", "--filter=blob:none", spec["official_repo"], str(destination))
        run("git", "-C", str(destination), "fetch", "origin", spec["revision"], "--depth", "1")
        run("git", "-C", str(destination), "checkout", "--detach", spec["revision"])
        run("git", "-C", str(destination), "submodule", "update", "--init", "--recursive")
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        print(f"{key}: {actual} ({destination})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
