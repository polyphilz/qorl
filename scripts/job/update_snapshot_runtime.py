#!/usr/bin/env python3
"""Refresh only a verified snapshot's Docker runtime identity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_PATH = REPOSITORY_ROOT / "docker/postgres/versions.json"


def inspect_image(reference: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", reference],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout)[0]


def write_atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def refresh_snapshot_runtime(
    path: Path,
    image: dict[str, Any],
    expected_benchmark_config_id: str,
) -> tuple[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    labels = image["Config"].get("Labels") or {}
    benchmark_config_id = labels.get("io.qorl.benchmark.config-id")
    if benchmark_config_id != expected_benchmark_config_id:
        raise RuntimeError(
            "refusing to record an image without the "
            f"{expected_benchmark_config_id} contract label"
        )

    old_image_id = manifest["image"]["id"]
    manifest["image"] = {
        "reference": manifest["image"]["reference"],
        "id": image["Id"],
        "architecture": image["Architecture"],
        "os": image["Os"],
        "benchmark_config_id": benchmark_config_id,
    }
    write_atomic(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return old_image_id, image["Id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    args = parser.parse_args()

    path = args.snapshot_manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    versions = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    expected_benchmark_config_id = versions["benchmark"]["config_id"]
    reference = manifest["image"]["reference"]
    image = inspect_image(reference)
    old_image_id, new_image_id = refresh_snapshot_runtime(
        path,
        image,
        expected_benchmark_config_id,
    )
    print(f"snapshot runtime identity refreshed: {old_image_id} -> {new_image_id}")


if __name__ == "__main__":
    main()
