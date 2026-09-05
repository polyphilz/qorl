"""Checksum-verified downloads and extraction of the pinned JOB source archive."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from qorl.util.hashing import sha256_file

TRANSFER_BLOCK_BYTES = 1024 * 1024
PROGRESS_INTERVAL_BYTES = 128 * TRANSFER_BLOCK_BYTES


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
    request = urllib.request.Request(url, headers={"User-Agent": "qorl-source-fetch/1"})
    downloaded = 0
    next_progress = PROGRESS_INTERVAL_BYTES
    try:
        with (
            urllib.request.urlopen(request) as response,
            temporary.open("xb") as output,
        ):
            while block := response.read(TRANSFER_BLOCK_BYTES):
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_progress:
                    print(f"downloaded {downloaded} bytes from {url}")
                    next_progress += PROGRESS_INTERVAL_BYTES
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


def extract_source(archive: Path, target: Path) -> None:
    if target.exists():
        return

    temporary = Path(tempfile.mkdtemp(prefix=".source.extract.", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as source:
            regular_members = [
                member for member in source.getmembers() if member.isfile()
            ]
            roots = {
                safe_member_path(member.name).parts[0] for member in regular_members
            }
            if len(roots) != 1:
                raise RuntimeError(
                    "JOB source archive must have exactly one root directory"
                )
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
                    shutil.copyfileobj(extracted, output, length=TRANSFER_BLOCK_BYTES)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"extracted JOB source: {target}")
