#!/usr/bin/env python3
"""Safely extract the recovered CEB SQL without unpickling qrep files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.utils.ceb import extract_sql_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPOSITORY_ROOT / "data/manifests/ceb-v1.json"
DEFAULT_FULL_SOURCE = REPOSITORY_ROOT / "data/raw/ceb-v1/source/imdb"
DEFAULT_UNIQUE_SOURCE = (
    REPOSITORY_ROOT / "data/raw/ceb-v1/source/imdb-unique-plans"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/ceb/ceb-v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(records: list[dict[str, Any]], key: str, path_key: str) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item[path_key]):
        digest.update(f"{record[key]}  {record[path_key]}\n".encode("utf-8"))
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    paths = sorted(root.glob("*/*.pkl"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise RuntimeError(f"no qrep files found under {root}")
    return paths


def expected_templates(specification: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(count)
        for name, count in specification["templates"].items()
    }


def validate_counts(
    paths: list[Path], root: Path, specification: dict[str, Any]
) -> None:
    counts = Counter(path.relative_to(root).parts[0] for path in paths)
    expected = expected_templates(specification)
    if len(paths) != specification["count"] or dict(sorted(counts.items())) != expected:
        raise RuntimeError(
            f"source query counts differ: expected={expected} "
            f"actual={dict(sorted(counts.items()))}"
        )


def extract_full_tree(
    root: Path,
    query_dir: Path,
    specification: dict[str, Any],
) -> list[dict[str, Any]]:
    paths = source_files(root)
    validate_counts(paths, root, specification)
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        template = relative.parts[0]
        source_name = relative.stem
        source_bytes = path.read_bytes()
        try:
            sql = extract_sql_bytes(source_bytes)
        except ValueError as error:
            raise RuntimeError(f"cannot extract SQL from {relative}: {error}") from error

        sql_path = Path("queries") / template / f"{source_name}.sql"
        destination = query_dir.parent / sql_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sql)
        records.append(
            {
                "source_id": f"{template}/{source_name}",
                "task_id": f"ceb-{template}-{source_name}",
                "template_id": f"ceb-{template}",
                "source_pickle_path": f"queries/imdb/{relative.as_posix()}",
                "source_pickle_sha256": sha256_bytes(source_bytes),
                "source_pickle_bytes": len(source_bytes),
                "sql_path": sql_path.as_posix(),
                "sql_sha256": sha256_bytes(sql),
                "sql_bytes": len(sql),
            }
        )
    return records


def build_unique_membership(
    root: Path,
    specification: dict[str, Any],
    full_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paths = source_files(root)
    validate_counts(paths, root, specification)
    by_source_id = {record["source_id"]: record for record in full_records}

    members: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        source_bytes = path.read_bytes()
        try:
            sql = extract_sql_bytes(source_bytes)
        except ValueError as error:
            raise RuntimeError(f"cannot extract SQL from {relative}: {error}") from error
        template_id = f"ceb-{relative.parts[0]}"
        sql_sha256 = sha256_bytes(sql)
        source_id = f"{relative.parts[0]}/{relative.stem}"
        full = by_source_id.get(source_id)
        if full is None or full["sql_sha256"] != sql_sha256:
            raise RuntimeError(
                f"unique-plan query {relative} does not match its full-workload query"
            )
        members.append(
            {
                "task_id": full["task_id"],
                "template_id": template_id,
                "sql_sha256": sql_sha256,
                "unique_source_pickle_path": (
                    f"queries/imdb-unique-plans/{relative.as_posix()}"
                ),
                "unique_source_pickle_sha256": sha256_bytes(source_bytes),
                "unique_source_pickle_bytes": len(source_bytes),
            }
        )
    if len({member["task_id"] for member in members}) != len(members):
        raise RuntimeError("unique-plan membership contains duplicate full-query IDs")
    return members


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build(
    source_manifest_path: Path,
    full_source: Path,
    unique_source: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build.", dir=output_dir.parent)
    )
    try:
        records = extract_full_tree(
            full_source,
            temporary / "queries",
            source_manifest["trees"]["full"],
        )
        members = build_unique_membership(
            unique_source,
            source_manifest["trees"]["unique_plans"],
            records,
        )
        counts = Counter(record["template_id"] for record in records)
        sources = {
            "schema_version": 1,
            "source_id": "ceb-v1-recovered-source",
            "source_manifest": {
                "path": str(source_manifest_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(source_manifest_path),
            },
            "extraction": {
                "method": "length-framed UTF-8 byte slicing",
                "pickle_imported_or_executed": False,
                "recognized_opcodes": ["SHORT_BINUNICODE", "BINUNICODE"],
            },
            "query_count": len(records),
            "template_count": len(counts),
            "template_query_counts": dict(sorted(counts.items())),
            "source_pickle_manifest_sha256": manifest_sha256(
                records, "source_pickle_sha256", "source_pickle_path"
            ),
            "sql_manifest_sha256": manifest_sha256(
                records, "sql_sha256", "sql_path"
            ),
            "queries": records,
        }
        unique_counts = Counter(member["template_id"] for member in members)
        unique = {
            "schema_version": 1,
            "membership_id": "ceb-v1-unique-plans",
            "definition": (
                "Recovered author-produced unique-plans subset mapped by exact "
                "template, source filename, and extracted SQL SHA-256 into the "
                "full workload"
            ),
            "query_count": len(members),
            "template_count": len(unique_counts),
            "template_query_counts": dict(sorted(unique_counts.items())),
            "sql_membership_manifest_sha256": manifest_sha256(
                members, "sql_sha256", "task_id"
            ),
            "source_pickle_manifest_sha256": manifest_sha256(
                members,
                "unique_source_pickle_sha256",
                "unique_source_pickle_path",
            ),
            "members": members,
        }
        write_json(temporary / "sources.json", sources)
        write_json(temporary / "unique-plans.json", unique)
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def check(source_manifest_path: Path, output_dir: Path) -> None:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sources = json.loads((output_dir / "sources.json").read_text(encoding="utf-8"))
    unique = json.loads((output_dir / "unique-plans.json").read_text(encoding="utf-8"))
    if sources["source_manifest"]["sha256"] != sha256_file(source_manifest_path):
        raise RuntimeError("checked-in sources reference a different source manifest")
    if sources["query_count"] != source_manifest["trees"]["full"]["count"]:
        raise RuntimeError("checked-in full query count differs from source manifest")
    if unique["query_count"] != source_manifest["trees"]["unique_plans"]["count"]:
        raise RuntimeError("checked-in unique query count differs from source manifest")

    records = sources["queries"]
    if len(records) != len({record["task_id"] for record in records}):
        raise RuntimeError("full source manifest contains duplicate task IDs")
    full_counts = Counter(record["template_id"].removeprefix("ceb-") for record in records)
    if dict(sorted(full_counts.items())) != expected_templates(
        source_manifest["trees"]["full"]
    ):
        raise RuntimeError("checked-in full template counts differ")
    expected_paths: set[str] = set()
    for record in records:
        path = output_dir / record["sql_path"]
        expected_paths.add(record["sql_path"])
        if path.stat().st_size != record["sql_bytes"] or sha256_file(path) != record["sql_sha256"]:
            raise RuntimeError(f"checked-in SQL differs from manifest: {path}")
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "queries").glob("*/*.sql")
    }
    if actual_paths != expected_paths:
        raise RuntimeError("checked-in SQL file set differs from sources.json")
    if sources["sql_manifest_sha256"] != manifest_sha256(
        records, "sql_sha256", "sql_path"
    ):
        raise RuntimeError("checked-in SQL manifest aggregate differs")
    if sources["source_pickle_manifest_sha256"] != manifest_sha256(
        records, "source_pickle_sha256", "source_pickle_path"
    ):
        raise RuntimeError("full source-pickle manifest aggregate differs")

    by_task = {record["task_id"]: record for record in records}
    task_ids = set(by_task)
    member_ids = [member["task_id"] for member in unique["members"]]
    if len(member_ids) != len(set(member_ids)) or not set(member_ids) <= task_ids:
        raise RuntimeError("unique-plan membership is not a subset of full queries")
    if any(
        member["sql_sha256"] != by_task[member["task_id"]]["sql_sha256"]
        for member in unique["members"]
    ):
        raise RuntimeError("unique-plan SQL differs from its full query")
    unique_counts = Counter(
        member["template_id"].removeprefix("ceb-")
        for member in unique["members"]
    )
    if dict(sorted(unique_counts.items())) != expected_templates(
        source_manifest["trees"]["unique_plans"]
    ):
        raise RuntimeError("checked-in unique template counts differ")
    if unique["sql_membership_manifest_sha256"] != manifest_sha256(
        unique["members"], "sql_sha256", "task_id"
    ):
        raise RuntimeError("unique-plan SQL membership aggregate differs")
    if unique["source_pickle_manifest_sha256"] != manifest_sha256(
        unique["members"],
        "unique_source_pickle_sha256",
        "unique_source_pickle_path",
    ):
        raise RuntimeError("unique source-pickle manifest aggregate differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--full-source", type=Path, default=DEFAULT_FULL_SOURCE)
    parser.add_argument("--unique-source", type=Path, default=DEFAULT_UNIQUE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args.source_manifest, args.output_dir)
        print("checked-in CEB SQL and source manifests verified")
    else:
        build(args.source_manifest, args.full_source, args.unique_source, args.output_dir)
        print("extracted CEB SQL without importing or executing pickle")


if __name__ == "__main__":
    main()
