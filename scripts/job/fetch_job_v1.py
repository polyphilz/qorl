#!/usr/bin/env python3
"""Fetch and verify the exact source inputs for the JOB v1 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/job/manifest.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data/raw/job-v1"


def sha256_stream(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return sha256_stream(source)


def verify_file(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"required file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != spec["bytes"]:
        raise RuntimeError(
            f"size mismatch for {path}: expected={spec['bytes']} actual={actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: "
            f"expected={spec['sha256']} actual={actual_sha256}"
        )


def download(url: str, target: Path, spec: dict[str, Any]) -> None:
    if target.exists():
        verify_file(target, spec)
        print(f"verified cached archive: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part.{os.getpid()}")
    request = urllib.request.Request(url, headers={"User-Agent": "qorl-job-v1-loader/1"})
    downloaded = 0
    next_progress = 128 * 1024 * 1024
    try:
        with urllib.request.urlopen(request) as response, temporary.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_progress:
                    print(f"downloaded {downloaded} bytes from {url}")
                    next_progress += 128 * 1024 * 1024
            output.flush()
            os.fsync(output.fileno())
        verify_file(temporary, spec)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"downloaded and verified: {target}")


def safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member: {name}")
    return path


def extract_dataset(
    archive: Path, target: Path, members: dict[str, dict[str, Any]]
) -> None:
    expected_names = set(members)
    if target.exists():
        verify_dataset_directory(target, members)
        print(f"verified extracted dataset: {target}")
        return

    temporary = Path(tempfile.mkdtemp(prefix=".imdb.extract.", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as source:
            archive_files = {
                safe_member_path(member.name).as_posix(): member
                for member in source.getmembers()
                if member.isfile()
            }
            if set(archive_files) != expected_names:
                missing = sorted(expected_names - set(archive_files))
                unexpected = sorted(set(archive_files) - expected_names)
                raise RuntimeError(
                    f"dataset archive member mismatch: missing={missing} "
                    f"unexpected={unexpected}"
                )
            for name in sorted(expected_names):
                member = archive_files[name]
                if member.size != members[name]["bytes"]:
                    raise RuntimeError(f"archive member size mismatch: {name}")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read archive member: {name}")
                output_path = temporary / name
                with output_path.open("xb") as output:
                    shutil.copyfileobj(extracted, output, length=1024 * 1024)
        verify_dataset_directory(temporary, members)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"extracted and verified dataset: {target}")


def verify_dataset_directory(
    target: Path, members: dict[str, dict[str, Any]]
) -> None:
    actual_names = {path.name for path in target.iterdir() if path.is_file()}
    expected_names = set(members)
    if actual_names != expected_names:
        raise RuntimeError(
            f"extracted dataset file mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name, spec in sorted(members.items()):
        verify_file(target / name, spec)


def extract_source(archive: Path, target: Path) -> None:
    if target.exists():
        return

    temporary = Path(tempfile.mkdtemp(prefix=".source.extract.", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as source:
            regular_members = [member for member in source.getmembers() if member.isfile()]
            roots = {
                safe_member_path(member.name).parts[0] for member in regular_members
            }
            if len(roots) != 1:
                raise RuntimeError("JOB source archive must have exactly one root directory")
            root = next(iter(roots))
            for member in regular_members:
                path = safe_member_path(member.name)
                relative = PurePosixPath(*path.parts[1:])
                if not relative.parts or len(relative.parts) != 1:
                    raise RuntimeError(f"unexpected JOB source path: {member.name}")
                if path.parts[0] != root:
                    raise RuntimeError(f"unexpected JOB source root: {member.name}")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                with (temporary / relative.name).open("xb") as output:
                    shutil.copyfileobj(extracted, output, length=1024 * 1024)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"extracted JOB source: {target}")


def query_manifest_sha256(source_dir: Path, pattern: str) -> tuple[int, str]:
    queries = sorted(source_dir.glob(pattern), key=lambda path: path.name)
    digest = hashlib.sha256()
    for query in queries:
        digest.update(f"{sha256_file(query)}  {query.name}\n".encode("ascii"))
    return len(queries), digest.hexdigest()


def verify_source_directory(target: Path, workload: dict[str, Any]) -> None:
    schema = target / workload["schema"]["path"]
    indexes = target / workload["indexes"]["path"]
    if sha256_file(schema) != workload["schema"]["sha256"]:
        raise RuntimeError(f"schema checksum mismatch: {schema}")
    if sha256_file(indexes) != workload["indexes"]["sha256"]:
        raise RuntimeError(f"index checksum mismatch: {indexes}")

    query_spec = workload["queries"]
    count, digest = query_manifest_sha256(target, query_spec["glob"])
    if count != query_spec["count"] or digest != query_spec["sha256_manifest"]:
        raise RuntimeError(
            "query-set mismatch: "
            f"expected_count={query_spec['count']} actual_count={count} "
            f"expected_sha256={query_spec['sha256_manifest']} actual_sha256={digest}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["fixture_id"] != "job-v1":
        raise RuntimeError("manifest is not the job-v1 fixture")

    args.raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = manifest["dataset"]
    dataset_archive = args.raw_dir / dataset["archive"]["filename"]
    download(dataset["source_url"], dataset_archive, dataset["archive"])
    extract_dataset(dataset_archive, args.raw_dir / "imdb", dataset["members"])

    workload = manifest["workload"]
    source_archive = args.raw_dir / workload["archive"]["filename"]
    download(workload["archive"]["url"], source_archive, workload["archive"])
    source_dir = args.raw_dir / "source"
    extract_source(source_archive, source_dir)
    verify_source_directory(source_dir, workload)

    print(
        f"job-v1 inputs verified: dataset_members={len(dataset['members'])} "
        f"queries={workload['queries']['count']}"
    )


if __name__ == "__main__":
    main()
