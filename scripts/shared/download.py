"""Download missing files without verifying their contents."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from urllib.response import addinfourl

BYTES_PER_KIB = 1024
TRANSFER_BLOCK_BYTES = 1024 * 1024
DEFAULT_PROGRESS_INTERVAL_IN_KIB = 128_000


def download(
    url: str,
    target: Path,
    *,
    print_progress: bool = False,
    progress_interval_in_kib: int = DEFAULT_PROGRESS_INTERVAL_IN_KIB,
) -> None:
    """Download a missing file, leaving existing targets untouched.

    progress_interval_in_kib is measured in KiB (1,024 bytes); the default is 125 MiB.
    Interrupted downloads leave no partial file. Callers verify completed files.
    """
    if progress_interval_in_kib <= 0:
        raise ValueError("progress_interval_in_kib must be greater than zero.")
    if target.exists():
        print(f"using existing file: {target}")
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
        temporary.replace(target)
    print(f"downloaded: {target}")
