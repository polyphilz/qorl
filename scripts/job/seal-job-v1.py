#!/usr/bin/env python3
"""Seal a cleanly stopped job-v1 PGDATA volume as a checksummed archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/manifests/job-v1.json"


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def inspect_container(container: str) -> dict[str, Any]:
    return json.loads(run(["docker", "container", "inspect", container]))[0]


def inspect_image(image_id: str) -> dict[str, Any]:
    return json.loads(run(["docker", "image", "inspect", image_id]))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-verification", required=True, type=Path)
    parser.add_argument("--environment-capture", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verification = json.loads(args.build_verification.read_text(encoding="utf-8"))
    if verification["phase"] != "build":
        raise RuntimeError("snapshot requires a build-phase database verification")
    if verification["source_manifest_sha256"] != sha256_file(args.manifest):
        raise RuntimeError("build verification does not match the current source manifest")

    container = inspect_container(args.container)
    if container["State"]["Running"]:
        raise RuntimeError("PostgreSQL must be stopped cleanly before snapshot sealing")
    if container["State"]["ExitCode"] != 0:
        raise RuntimeError(
            f"PostgreSQL container exited unsuccessfully: {container['State']['ExitCode']}"
        )

    environment = dict(
        item.split("=", 1)
        for item in container["Config"]["Env"]
        if "=" in item
    )
    pgdata = PurePosixPath(environment["PGDATA"])
    data_mounts = []
    for mount in container["Mounts"]:
        destination = PurePosixPath(mount["Destination"])
        if pgdata == destination or destination in pgdata.parents:
            data_mounts.append(mount)
    if len(data_mounts) != 1 or data_mounts[0]["Type"] != "volume":
        raise RuntimeError("container must have one named PGDATA volume")
    data_mount = data_mounts[0]
    volume_name = data_mount["Name"]
    pgdata_relative_path = pgdata.relative_to(data_mount["Destination"])

    image_reference = container["Config"]["Image"]
    image_id = container["Image"]
    if image_reference != manifest["database"]["image_reference"]:
        raise RuntimeError(
            f"snapshot image mismatch: expected={manifest['database']['image_reference']} "
            f"actual={image_reference}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "job-v1.snapshot.tar.gz"
    partial = args.output_dir / ".job-v1.snapshot.tar.gz.part"
    snapshot_manifest = args.output_dir / "job-v1.snapshot.json"
    for path in (archive, partial, snapshot_manifest):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite snapshot output: {path}")

    uid = os.getuid()
    gid = os.getgid()
    archive_script = f"""
set -Eeuo pipefail
partial=/output/{partial.name}
source=/source/$1
trap 'chown {uid}:{gid} "$partial" 2>/dev/null || true' EXIT
test -f "$source/PG_VERSION"
test ! -e "$source/postmaster.pid"
tar \
    --create \
    --directory="$source" \
    --sort=name \
    --mtime=@0 \
    --owner=999 \
    --group=999 \
    --numeric-owner \
    --format=gnu \
    . \
    | gzip --no-name --fast > "$partial"
test -s "$partial"
"""

    try:
        run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--volume",
                f"{volume_name}:/source:ro",
                "--volume",
                f"{args.output_dir.resolve()}:/output",
                "--entrypoint",
                "bash",
                image_id,
                "-Eeuo",
                "pipefail",
                "-c",
                archive_script,
                "qorl-seal",
                str(pgdata_relative_path),
            ]
        )
        partial.replace(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    archive_sha256 = sha256_file(archive)
    image = inspect_image(image_id)
    environment_record = None
    if args.environment_capture:
        environment_record = {
            "filename": args.environment_capture.name,
            "bytes": args.environment_capture.stat().st_size,
            "sha256": sha256_file(args.environment_capture),
        }

    result = {
        "schema_version": 1,
        "snapshot_id": f"job-v1-sha256-{archive_sha256[:16]}",
        "fixture_id": manifest["fixture_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "qorl-pgdata-tar-gzip-v1",
        "archive": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        },
        "source_manifest": {
            "filename": args.manifest.name,
            "sha256": sha256_file(args.manifest),
        },
        "build_verification": {
            "filename": args.build_verification.name,
            "bytes": args.build_verification.stat().st_size,
            "sha256": sha256_file(args.build_verification),
        },
        "environment_capture": environment_record,
        "postgresql": {
            "system_identifier": verification["database"]["identity"][
                "system_identifier"
            ],
            "server_version_num": verification["database"]["identity"][
                "server_version_num"
            ],
            "pgdata": str(pgdata),
            "volume_destination": data_mount["Destination"],
            "pgdata_volume_relative_path": str(pgdata_relative_path),
            "clean_shutdown": True,
        },
        "image": {
            "reference": image_reference,
            "id": image_id,
            "architecture": image["Architecture"],
            "os": image["Os"],
            "benchmark_config_id": (image["Config"].get("Labels") or {}).get(
                "io.qorl.benchmark.config-id"
            ),
        },
        "source_volume": volume_name,
        "normalization": {
            "tar_order": "name",
            "mtime": 0,
            "uid": 999,
            "gid": 999,
            "gzip_original_name_and_timestamp": False,
        },
    }
    write_atomic(snapshot_manifest, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"sealed {result['snapshot_id']}: "
        f"archive={archive} bytes={archive.stat().st_size}"
    )


if __name__ == "__main__":
    main()
