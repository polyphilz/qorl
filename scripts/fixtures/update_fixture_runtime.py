#!/usr/bin/env python3
"""Refresh only a verified manifest's Docker runtime identity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def inspect_image(reference: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", reference],
        text=True,
        capture_output=True,
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


def refresh_fixture_runtime(
    path: Path,
    image: dict[str, Any],
) -> tuple[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    old_image_id = manifest["image"]["id"]
    manifest["image"] = {
        "reference": manifest["image"]["reference"],
        "id": image["Id"],
        "architecture": image["Architecture"],
        "os": image["Os"],
    }
    write_atomic(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return old_image_id, image["Id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-manifest", required=True, type=Path)
    args = parser.parse_args()

    path = args.archive_manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    reference = manifest["image"]["reference"]
    image = inspect_image(reference)
    old_image_id, new_image_id = refresh_fixture_runtime(
        path,
        image,
    )
    print(f"manifest runtime identity refreshed: {old_image_id} -> {new_image_id}")


if __name__ == "__main__":
    main()
