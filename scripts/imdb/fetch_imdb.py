"""Fetch IMDb rows and schema/index definitions from the pinned sources."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from qorl.util.hashing import sha256_file

from scripts.shared.source_archive import (  # noqa: E402
    TRANSFER_BLOCK_BYTES,
    download,
    extract_source,
    safe_member_path,
    verify_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_MANIFEST = REPOSITORY_ROOT / "scripts/imdb/imdb-metadata.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data/raw"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "benchmarks/raw/job"


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
                    shutil.copyfileobj(extracted, output, length=TRANSFER_BLOCK_BYTES)
        verify_dataset_directory(temporary, members)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"extracted and verified dataset: {target}")


def verify_dataset_directory(target: Path, members: dict[str, dict[str, Any]]) -> None:
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


def verify_schema_directory(target: Path, source: dict[str, Any]) -> None:
    for kind in ("schema", "indexes"):
        spec = source[kind]
        path = target / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"{kind} checksum mismatch: {path}")


def verify_inputs(repository: Path) -> None:
    manifest = json.loads(
        (repository / "scripts/imdb/imdb-metadata.json").read_text(encoding="utf-8")
    )
    verify_dataset_directory(
        repository / "data/raw/tables", manifest["dataset"]["members"]
    )
    verify_schema_directory(
        repository / "benchmarks/raw/job/source", manifest["schema_source"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["fixture_id"] != "imdb":
        raise RuntimeError("manifest is not the IMDb fixture")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = manifest["dataset"]
    archive = args.raw_dir / dataset["archive"]["filename"]
    download(dataset["source_url"], archive, dataset["archive"])
    extract_dataset(archive, args.raw_dir / "tables", dataset["members"])

    source = manifest["schema_source"]
    source_archive = args.source_dir / source["archive"]["filename"]
    download(source["archive"]["url"], source_archive, source["archive"])
    extract_source(source_archive, args.source_dir / "source")
    verify_schema_directory(args.source_dir / "source", source)
    print(f"IMDb inputs verified: dataset_members={len(dataset['members'])}")


if __name__ == "__main__":
    main()
