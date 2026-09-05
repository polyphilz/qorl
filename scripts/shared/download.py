"""Download files and check their expected size and SHA-256."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from urllib.response import addinfourl

from qorl.util.hashing import sha256_file

BYTES_PER_KIB = 1024
TRANSFER_BLOCK_BYTES = 1024 * 1024
DEFAULT_PROGRESS_INTERVAL_IN_KIB = 128_000


def verify_file(
    target: Path, expected_size_in_bytes: int, expected_checksum: str
) -> None:
    if not target.is_file():
        raise RuntimeError(
            f"Expected a regular file at {target}; it is missing or is not a file."
        )
    actual_size = target.stat().st_size
    if actual_size != expected_size_in_bytes:
        raise RuntimeError(
            f"Size mismatch for {target}: expected {expected_size_in_bytes} bytes, "
            f"got {actual_size} bytes."
        )
    actual_checksum = sha256_file(target)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"SHA-256 mismatch for {target}: expected {expected_checksum}, "
            f"got {actual_checksum}."
        )


def download(
    url: str,
    target: Path,
    expected_size_in_bytes: int,
    expected_checksum: str,
    *,
    print_progress: bool = False,
    progress_interval_in_kib: int = DEFAULT_PROGRESS_INTERVAL_IN_KIB,
) -> None:
    """Verify a cached file or download and verify it before publishing it.

    progress_interval_in_kib is measured in KiB (1,024 bytes); the default is 125 MiB.
    Invalid cached files are left untouched. Failed downloads leave no partial file.
    """
    if progress_interval_in_kib <= 0:
        raise ValueError("progress_interval_in_kib must be greater than zero.")
    if target.exists():
        verify_file(target, expected_size_in_bytes, expected_checksum)
        print(f"verified cached file: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "qorl-source-fetch/1"})
    interval_bytes = progress_interval_in_kib * BYTES_PER_KIB
    next_progress = interval_bytes
    downloaded = 0
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.", dir=target.parent
    ) as directory:
        temporary = Path(directory) / target.name
        raw_response: object
        with (
            urllib.request.urlopen(request) as raw_response,
            temporary.open("xb") as output,
        ):
            if not isinstance(raw_response, (HTTPResponse, addinfourl)):
                raise TypeError(
                    "Download response must be an HTTP or binary URL response."
                )
            response = raw_response
            while block := response.read(TRANSFER_BLOCK_BYTES):
                output.write(block)
                downloaded += len(block)
                if print_progress and downloaded >= next_progress:
                    print(f"downloaded {downloaded} bytes from {url}", flush=True)
                    next_progress = (downloaded // interval_bytes + 1) * interval_bytes
            output.flush()
            os.fsync(output.fileno())
        verify_file(temporary, expected_size_in_bytes, expected_checksum)
        temporary.replace(target)
    print(f"downloaded and verified: {target}")
