#!/usr/bin/env python3
"""Fetch and safely extract the pinned recovered CEB query representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from qorl.util.hashing import sha256_stream

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/ceb/manifest.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data/raw/ceb-v1"
TREE_NAMES = {"full": "imdb", "unique_plans": "imdb-unique-plans"}
QUERY_ARCHIVE_PATH_PARTS = 5


def verify_file(path: Path, specification: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"required archive is missing: {path}")
    if path.stat().st_size != specification["bytes"]:
        raise RuntimeError(f"archive byte length differs: {path}")
    with path.open("rb") as source:
        actual = sha256_stream(source)
    if actual != specification["sha256"]:
        raise RuntimeError(
            f"archive SHA-256 differs: expected={specification['sha256']} "
            f"actual={actual}"
        )


def download(url: str, target: Path, specification: dict[str, Any]) -> None:
    if target.exists():
        verify_file(target, specification)
        print(f"verified cached CEB archive: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part.{os.getpid()}")
    request = urllib.request.Request(url, headers={"User-Agent": "qorl-ceb-v1/1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        verify_file(partial, specification)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"downloaded and verified CEB archive: {target}")


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member path: {name}")
    return path


def selected_members(
    archive: tarfile.TarFile, manifest: dict[str, Any]
) -> dict[str, tarfile.TarInfo]:
    archive_members = archive.getmembers()
    archive_specification = manifest["provenance"]["immutable_archive"]
    if len(archive_members) != archive_specification["members"]:
        raise RuntimeError("archive member count differs")
    if (
        sum(member.isfile() for member in archive_members)
        != archive_specification["regular_files"]
    ):
        raise RuntimeError("archive regular-file count differs")
    roots: set[str] = set()
    selected: dict[str, tarfile.TarInfo] = {}
    counts: dict[str, Counter[str]] = {key: Counter() for key in TREE_NAMES}
    for member in archive_members:
        path = safe_path(member.name)
        if path.parts:
            roots.add(path.parts[0])
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive links are forbidden: {member.name}")
        if not member.isfile():
            if not member.isdir():
                raise RuntimeError(f"unsupported archive member: {member.name}")
            continue
        if len(path.parts) != QUERY_ARCHIVE_PATH_PARTS or path.parts[1] != "queries":
            continue
        tree = next(
            (key for key, name in TREE_NAMES.items() if path.parts[2] == name),
            None,
        )
        if tree is None or path.suffix != ".pkl":
            continue
        template = path.parts[3]
        relative = PurePosixPath(path.parts[2], template, path.name).as_posix()
        if relative in selected:
            raise RuntimeError(f"duplicate archive member: {member.name}")
        selected[relative] = member
        counts[tree][template] += 1

    if len(roots) != 1:
        raise RuntimeError(f"archive must have one root directory: {sorted(roots)}")
    for key in TREE_NAMES:
        specification = manifest["trees"][key]
        expected = {
            template: int(count)
            for template, count in specification["templates"].items()
        }
        if sum(counts[key].values()) != specification["count"]:
            raise RuntimeError(f"archive {key} query count differs")
        if dict(sorted(counts[key].items())) != expected:
            raise RuntimeError(f"archive {key} template counts differ")
    return selected


def stream_sha256(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def verify_extracted(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    target: Path,
) -> None:
    actual = {path.relative_to(target).as_posix() for path in target.glob("*/*/*.pkl")}
    if actual != set(members):
        raise RuntimeError("extracted CEB file set differs from the pinned archive")
    for relative, member in sorted(members.items()):
        archived = archive.extractfile(member)
        if archived is None:
            raise RuntimeError(f"cannot read archive member: {member.name}")
        archive_size, archive_sha256 = stream_sha256(archived)
        path = target / relative
        with path.open("rb") as source:
            file_size, file_sha256 = stream_sha256(source)
        if (file_size, file_sha256) != (archive_size, archive_sha256):
            raise RuntimeError(f"extracted CEB file differs: {relative}")


def extract(
    archive_path: Path,
    target: Path,
    manifest: dict[str, Any],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = selected_members(archive, manifest)
        if target.exists():
            verify_extracted(archive, members, target)
            print(f"verified extracted CEB trees: {target}")
            return

        temporary = Path(
            tempfile.mkdtemp(prefix=".ceb-source.extract.", dir=target.parent)
        )
        try:
            for relative, member in sorted(members.items()):
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                output = temporary / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            verify_extracted(archive, members, temporary)
            temporary.replace(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    print(f"safely extracted CEB query trees: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["workload_id"] != "ceb-v1":
        raise RuntimeError("manifest is not ceb-v1")
    specification = manifest["provenance"]["immutable_archive"]
    archive = args.raw_dir / specification["filename"]
    download(specification["url"], archive, specification)
    extract(archive, args.raw_dir / "source", manifest)
    print("ceb-v1 recovered source inputs verified")


if __name__ == "__main__":
    main()
