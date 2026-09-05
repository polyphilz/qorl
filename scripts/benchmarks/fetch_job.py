"""Fetch and verify JOB queries, without downloading IMDb table data."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from qorl.util.hashing import sha256_bytes, sha256_file
from scripts.shared.download import download

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/job/manifest.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "benchmarks/raw/job"
TRANSFER_BLOCK_BYTES = 1024 * 1024


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["workload_id"] != "job":
        raise RuntimeError("manifest is not the JOB workload")
    archive_specification = manifest["source"]["archive"]
    archive = args.raw_dir / archive_specification["filename"]
    download(
        archive_specification["url"],
        archive,
        expected_size_in_bytes=archive_specification["bytes"],
        expected_checksum=archive_specification["sha256"],
    )
    source = args.raw_dir / "source"
    extract_source(archive, source)
    specification = manifest["queries"]
    queries = sorted(source.glob(specification["glob"]))
    query_checksum = sha256_bytes(
        "".join(
            f"{sha256_file(query)}  {query.name}\n"
            for query in sorted(queries, key=lambda path: path.name)
        ).encode("ascii")
    )
    if (
        len(queries) != specification["count"]
        or query_checksum != specification["sha256_manifest"]
    ):
        raise RuntimeError("JOB query count or checksum differs from the manifest")
    print(f"JOB queries verified: {len(queries)}")


if __name__ == "__main__":
    main()
