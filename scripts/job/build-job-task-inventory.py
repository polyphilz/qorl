#!/usr/bin/env python3
"""Build or verify the checked-in JOB task inventory."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPOSITORY_ROOT / "data/manifests/job-v1.json"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "data/raw/job-v1/source"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data/job/job-v1"

QUERY_NAME = re.compile(r"^(?P<template>[1-9][0-9]*)(?P<variant>[a-z])\.sql$")
TABLE_TERM = re.compile(
    r"^\s*(?P<table>[a-z_][a-z0-9_]*)\s+(?:AS\s+)?"
    r"(?P<alias>[a-z_][a-z0-9_]*)\s*$",
    re.IGNORECASE,
)
JOIN_PREDICATE = re.compile(
    r"\b(?P<left_alias>[a-z_][a-z0-9_]*)\."
    r"(?P<left_column>[a-z_][a-z0-9_]*)\s*=\s*"
    r"(?P<right_alias>[a-z_][a-z0-9_]*)\."
    r"(?P<right_column>[a-z_][a-z0-9_]*)\b",
    re.IGNORECASE,
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_join_graph_sha256(
    aliases: dict[str, str],
    join_predicates: set[tuple[str, str, str, str]],
) -> str:
    aliases_by_table: dict[str, list[str]] = {}
    for alias, table in aliases.items():
        aliases_by_table.setdefault(table, []).append(alias)

    permutation_count = math.prod(
        math.factorial(len(table_aliases))
        for table_aliases in aliases_by_table.values()
    )
    if permutation_count > 1_000_000:
        raise RuntimeError(
            f"join graph has too many alias permutations: {permutation_count}"
        )

    assignment_groups: list[list[dict[str, str]]] = []
    for table, table_aliases in sorted(aliases_by_table.items()):
        table_aliases = sorted(table_aliases)
        assignments: list[dict[str, str]] = []
        for ordering in itertools.permutations(table_aliases):
            assignments.append(
                {
                    alias: table if len(ordering) == 1 else f"{table}#{index}"
                    for index, alias in enumerate(ordering, start=1)
                }
            )
        assignment_groups.append(assignments)

    canonical_encodings: list[str] = []
    for assignment_group in itertools.product(*assignment_groups):
        node_names = {
            alias: node_name
            for assignment in assignment_group
            for alias, node_name in assignment.items()
        }
        canonical_edges: set[str] = set()
        for left_alias, left_column, right_alias, right_column in join_predicates:
            left = f"{node_names[left_alias]}.{left_column}"
            right = f"{node_names[right_alias]}.{right_column}"
            first, second = sorted((left, right))
            canonical_edges.add(f"{first}={second}")
        canonical_encodings.append(
            json.dumps(
                {
                    "relations": sorted(node_names.values()),
                    "join_edges": sorted(canonical_edges),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    return sha256_bytes(min(canonical_encodings).encode("utf-8"))


def natural_query_key(path: Path) -> tuple[int, str]:
    match = QUERY_NAME.fullmatch(path.name)
    if not match:
        raise RuntimeError(f"invalid JOB query filename: {path.name}")
    return int(match["template"]), match["variant"]


def extract_structure(sql: str, query_name: str) -> dict[str, Any]:
    from_match = re.search(r"\bFROM\s+(.*?)\s+WHERE\b", sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        raise RuntimeError(f"cannot find FROM/WHERE clauses in {query_name}")

    aliases: dict[str, str] = {}
    for term in from_match.group(1).split(","):
        match = TABLE_TERM.fullmatch(term)
        if not match:
            raise RuntimeError(f"cannot parse FROM term in {query_name}: {term!r}")
        table = match["table"].lower()
        alias = match["alias"].lower()
        if alias in aliases:
            raise RuntimeError(f"duplicate table alias in {query_name}: {alias}")
        aliases[alias] = table

    join_edges: set[str] = set()
    join_predicates: set[tuple[str, str, str, str]] = set()
    for match in JOIN_PREDICATE.finditer(sql):
        left_alias = match["left_alias"].lower()
        right_alias = match["right_alias"].lower()
        if left_alias not in aliases or right_alias not in aliases:
            continue
        if left_alias == right_alias:
            continue
        left_column = match["left_column"].lower()
        right_column = match["right_column"].lower()
        left = (
            f"{left_alias}:{aliases[left_alias]}."
            f"{left_column}"
        )
        right = (
            f"{right_alias}:{aliases[right_alias]}."
            f"{right_column}"
        )
        first, second = sorted((left, right))
        join_edges.add(f"{first}={second}")
        first_predicate, second_predicate = sorted(
            (
                (left_alias, left_column),
                (right_alias, right_column),
            )
        )
        join_predicates.add((*first_predicate, *second_predicate))

    relations = [
        {"alias": alias, "table": table}
        for alias, table in sorted(aliases.items())
    ]
    tables = sorted(set(aliases.values()))
    edges = sorted(join_edges)
    if len(relations) < 2 or not edges:
        raise RuntimeError(f"query has no usable join graph: {query_name}")

    adjacency = {alias: set() for alias in aliases}
    for left_alias, _left_column, right_alias, _right_column in join_predicates:
        adjacency[left_alias].add(right_alias)
        adjacency[right_alias].add(left_alias)
    visited: set[str] = set()
    frontier = [next(iter(aliases))]
    while frontier:
        alias = frontier.pop()
        if alias in visited:
            continue
        visited.add(alias)
        frontier.extend(adjacency[alias] - visited)
    if visited != set(aliases):
        raise RuntimeError(
            f"query join graph is disconnected: {query_name} "
            f"unreachable={sorted(set(aliases) - visited)}"
        )

    return {
        "tables": tables,
        "relations": relations,
        "join_edges": edges,
        "table_count": len(tables),
        "relation_count": len(relations),
        "join_predicate_count": len(edges),
        "join_graph_sha256": canonical_join_graph_sha256(
            aliases, join_predicates
        ),
    }


def query_manifest_sha256(queries: list[Path]) -> str:
    digest = hashlib.sha256()
    for query in sorted(queries, key=lambda path: path.name):
        digest.update(f"{sha256_file(query)}  {query.name}\n".encode("ascii"))
    return digest.hexdigest()


def build_tasks(source_dir: Path, query_glob: str) -> tuple[list[dict[str, Any]], list[Path]]:
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


def database_identity(
    snapshot_manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    if snapshot["source_manifest"]["sha256"] != source_manifest_sha256:
        raise RuntimeError("snapshot and JOB source manifests do not match")
    return {
        "fixture_id": snapshot["fixture_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_archive_sha256": snapshot["archive"]["sha256"],
        "postgres_image_id": snapshot["image"]["id"],
        "postgres_system_identifier": snapshot["postgresql"]["system_identifier"],
    }


def build_inventory(
    source_manifest_path: Path,
    source_dir: Path,
    snapshot_manifest_path: Path | None,
    existing_database: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    query_spec = source_manifest["workload"]["queries"]
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
    if snapshot_manifest_path:
        database = database_identity(snapshot_manifest_path, source_manifest_sha256)
    elif existing_database:
        database = existing_database
    else:
        raise RuntimeError("a snapshot manifest is required when creating the inventory")

    template_ids = sorted({task["template_id"] for task in tasks})
    inventory = {
        "schema_version": 1,
        "inventory_id": "job-v1-tasks-v1",
        "role": "held_out_test",
        "training_allowed": False,
        "tuning_allowed": False,
        "task_count": len(tasks),
        "template_count": len(template_ids),
        "database": database,
        "query_source": {
            "repository_url": source_manifest["workload"]["repository_url"],
            "commit": source_manifest["workload"]["commit"],
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


def write_inventory(output_dir: Path, inventory: dict[str, Any], queries: list[Path]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
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
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def check_inventory(
    output_dir: Path,
    source_manifest_path: Path,
    source_dir: Path,
    snapshot_manifest_path: Path | None,
) -> None:
    inventory_path = output_dir / "tasks.json"
    existing = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected, source_queries = build_inventory(
        source_manifest_path,
        source_dir,
        snapshot_manifest_path,
        existing_database=existing["database"],
    )
    if existing != expected:
        raise RuntimeError("checked-in tasks.json does not match the pinned JOB sources")

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
            raise RuntimeError(f"checked-in SQL differs from source: {source_query.name}")


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
    parser.add_argument("--snapshot-manifest", type=Path)
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
            args.snapshot_manifest,
        )
        print("checked-in job-v1 task inventory verification passed")
        return

    inventory, queries = build_inventory(
        args.source_manifest,
        source_dir,
        args.snapshot_manifest,
    )
    write_inventory(args.output_dir, inventory, queries)
    print(
        f"built job-v1 task inventory: "
        f"tasks={inventory['task_count']} templates={inventory['template_count']}"
    )


if __name__ == "__main__":
    main()
