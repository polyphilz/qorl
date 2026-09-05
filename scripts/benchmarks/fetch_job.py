"""Fetch and verify JOB queries, without downloading IMDb table data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmarks.build_job_task_inventory import query_manifest_sha256
from scripts.source_archive import download, extract_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/job/manifest.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "benchmarks/raw/job"


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
    download(archive_specification["url"], archive, archive_specification)
    source = args.raw_dir / "source"
    extract_source(archive, source)
    specification = manifest["queries"]
    queries = sorted(source.glob(specification["glob"]))
    if (
        len(queries) != specification["count"]
        or query_manifest_sha256(queries) != specification["sha256_manifest"]
    ):
        raise RuntimeError("JOB query count or checksum differs from the manifest")
    print(f"JOB queries verified: {len(queries)}")


if __name__ == "__main__":
    main()
