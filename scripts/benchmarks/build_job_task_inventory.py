#!/usr/bin/env python3
"""Build or verify the checked-in JOB task inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from qorl.util.hashing import sha256_bytes, sha256_file
from qorl.workload.query_structure import (
    extract_join_structure as extract_structure,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPOSITORY_ROOT / "benchmarks/job/manifest.json"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "benchmarks/raw/job/source"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmarks/job"
INVENTORY_SCHEMA_VERSION = 3

QUERY_NAME = re.compile(r"^(?P<template>[1-9][0-9]*)(?P<variant>[a-z])\.sql$")


def natural_query_key(path: Path) -> tuple[int, str]:
    match = QUERY_NAME.fullmatch(path.name)
    if not match:
        raise RuntimeError(f"invalid JOB query filename: {path.name}")
    return int(match["template"]), match["variant"]


def query_manifest_sha256(queries: list[Path]) -> str:
    digest = hashlib.sha256()
    for query in sorted(queries, key=lambda path: path.name):
        digest.update(f"{sha256_file(query)}  {query.name}\n".encode("ascii"))
    return digest.hexdigest()


def build_tasks(
    source_dir: Path, query_glob: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    queries = sorted(source_dir.glob(query_glob), key=natural_query_key)
    tasks: list[dict[str, Any]] = []
    for query in queries:
        match = QUERY_NAME.fullmatch(query.name)
        assert match is not None
        template_number = int(match["template"])
        variant = match["variant"]
        content = query.read_bytes()
        try:
            sql = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"query is not UTF-8: {query}") from error
        if not sql.lstrip().upper().startswith("SELECT"):
            raise RuntimeError(f"query is not a SELECT statement: {query.name}")

        task = {
            "task_id": f"job-{template_number:02d}{variant}",
            "template_id": f"job-{template_number:02d}",
            "sql_path": f"queries/{query.name}",
            "sql_sha256": sha256_bytes(content),
        }
        task.update(extract_structure(sql, query.name))
        tasks.append(task)
    return tasks, queries


def build_inventory(
    source_manifest_path: Path,
    source_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    query_spec = source_manifest["queries"]
    tasks, queries = build_tasks(source_dir, query_spec["glob"])

    if len(tasks) != query_spec["count"]:
        raise RuntimeError(
            f"query count mismatch: expected={query_spec['count']} actual={len(tasks)}"
        )
    aggregate_sha256 = query_manifest_sha256(queries)
    if aggregate_sha256 != query_spec["sha256_manifest"]:
        raise RuntimeError(
            "query source checksum mismatch: "
            f"expected={query_spec['sha256_manifest']} actual={aggregate_sha256}"
        )

    source_manifest_sha256 = sha256_file(source_manifest_path)

    template_ids = sorted({task["template_id"] for task in tasks})
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": "job",
        "task_count": len(tasks),
        "template_count": len(template_ids),
        "fixture_id": source_manifest["fixture_id"],
        "query_source": {
            "repository_url": source_manifest["source"]["repository_url"],
            "commit": source_manifest["source"]["commit"],
            "source_manifest_sha256": source_manifest_sha256,
            "query_manifest_sha256": aggregate_sha256,
        },
        "structure_extraction": {
            "version": 1,
            "join_graph_definition": "sorted relation aliases with source tables plus sorted cross-relation column-equality predicates",
            "cross_corpus_comparison": "join_graph_sha256 is alias-independent and canonicalizes repeated instances of the same table by exhaustive table-colored relabeling",
        },
        "tasks": tasks,
    }
    return inventory, queries


def write_inventory(
    output_dir: Path, inventory: dict[str, Any], queries: list[Path]
) -> None:
    generated = ("queries", "tasks.json")
    if any((output_dir / name).exists() for name in generated):
        raise RuntimeError(
            f"refusing to overwrite existing JOB queries or inventory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build.", dir=output_dir.parent)
    )
    try:
        query_dir = temporary / "queries"
        query_dir.mkdir()
        for query in queries:
            shutil.copyfile(query, query_dir / query.name)
        (temporary / "tasks.json").write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(exist_ok=True)
        for name in generated:
            (temporary / name).replace(output_dir / name)
        temporary.rmdir()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def check_inventory(
    output_dir: Path,
    source_manifest_path: Path,
    source_dir: Path,
) -> None:
    inventory_path = output_dir / "tasks.json"
    existing = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected, source_queries = build_inventory(
        source_manifest_path,
        source_dir,
    )
    if existing != expected:
        raise RuntimeError(
            "checked-in tasks.json does not match the pinned JOB sources"
        )

    expected_names = {query.name for query in source_queries}
    query_dir = output_dir / "queries"
    actual_names = {path.name for path in query_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"checked-in query file set mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for source_query in source_queries:
        checked_in = query_dir / source_query.name
        if source_query.read_bytes() != checked_in.read_bytes():
            raise RuntimeError(
                f"checked-in SQL differs from source: {source_query.name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "directory containing pinned upstream JOB queries; defaults to the "
            "raw source directory when building and the checked-in query directory "
            "when checking"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir
    if source_dir is None:
        source_dir = args.output_dir / "queries" if args.check else DEFAULT_SOURCE_DIR

    if args.check:
        check_inventory(
            args.output_dir,
            args.source_manifest,
            source_dir,
        )
        print("checked-in JOB task inventory verification passed")
        return

    inventory, queries = build_inventory(
        args.source_manifest,
        source_dir,
    )
    write_inventory(args.output_dir, inventory, queries)
    print(
        f"built JOB task inventory: "
        f"tasks={inventory['task_count']} templates={inventory['template_count']}"
    )


if __name__ == "__main__":
    main()
