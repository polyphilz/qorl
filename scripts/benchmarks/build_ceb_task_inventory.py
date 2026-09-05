#!/usr/bin/env python3
"""Build or verify the CEB task inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qorl.db.fixture import archive_data_identity
from qorl.util.hashing import sha256_bytes, sha256_file
from qorl.workload.query_structure import (
    extract_join_structure,
    task_join_fingerprints,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CEB_DIR = REPOSITORY_ROOT / "benchmarks/ceb"
DEFAULT_ARCHIVE_MANIFEST = REPOSITORY_ROOT / "imdb/archive.json"
INVENTORY_SCHEMA_VERSION = 3


def build_inventory(ceb_dir: Path, archive_manifest_path: Path) -> dict[str, Any]:
    provenance_dir = ceb_dir / "provenance"
    sources_path = provenance_dir / "sources.json"
    unique_path = provenance_dir / "unique-plans.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    unique = json.loads(unique_path.read_text(encoding="utf-8"))
    manifest = json.loads((ceb_dir / "manifest.json").read_text(encoding="utf-8"))
    database = archive_data_identity(archive_manifest_path, manifest["fixture_id"])

    source_templates = set(sources["template_query_counts"])
    source_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources["queries"]:
        source_groups[(source["template_id"], source["sql_sha256"])].append(source)
    counts = Counter(template for template, _sql_sha256 in source_groups)
    if set(counts) != source_templates:
        raise RuntimeError("source query records and template counts differ")
    unique_ids = {member["task_id"] for member in unique["members"]}

    tasks: list[dict[str, Any]] = []
    for (_template_id, _sql_sha256), group in sorted(source_groups.items()):
        group.sort(key=lambda source: source["source_id"])
        source = group[0]
        path = ceb_dir / source["sql_path"]
        content = path.read_bytes()
        if sha256_bytes(content) != source["sql_sha256"]:
            raise RuntimeError(f"SQL checksum differs from sources.json: {path}")
        try:
            sql = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"query is not UTF-8: {path}") from error
        structure = extract_join_structure(sql, source["source_id"])
        graph_sha256, topology_sha256 = task_join_fingerprints(structure)
        if graph_sha256 != structure["join_graph_sha256"]:
            raise RuntimeError(f"join fingerprint disagreement: {path}")
        task: dict[str, Any] = {
            "task_id": source["task_id"],
            "template_id": source["template_id"],
            "in_author_unique_plans_subset": any(
                member["task_id"] in unique_ids for member in group
            ),
            "source_id": source["source_id"],
            "source_pickle_sha256": source["source_pickle_sha256"],
            "sql_path": source["sql_path"],
            "sql_sha256": source["sql_sha256"],
            **structure,
            "join_topology_sha256": topology_sha256,
        }
        if len(group) > 1:
            task["duplicate_source_ids"] = [member["source_id"] for member in group[1:]]
        tasks.append(task)

    if len(tasks) != sum(counts.values()):
        raise RuntimeError("query count differs from built task count")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": "ceb",
        "task_count": len(tasks),
        "template_count": len(source_templates),
        "database": database,
        "query_source": {
            "source_id": sources["source_id"],
            "source_manifest": sources["source_manifest"],
            "sources_sha256": sha256_file(sources_path),
            "sql_manifest_sha256": sources["sql_manifest_sha256"],
            "source_pickle_manifest_sha256": sources["source_pickle_manifest_sha256"],
            "author_unique_plans_membership_sha256": sha256_file(unique_path),
        },
        "exact_sql_deduplication": {
            "unit": "template ID plus extracted SQL SHA-256",
            "source_representation_count": sum(
                len(group) for group in source_groups.values()
            ),
            "distinct_task_count": len(source_groups),
            "duplicate_representation_count": sum(
                len(group) - 1 for group in source_groups.values()
            ),
            "duplicate_group_count": sum(
                len(group) > 1 for group in source_groups.values()
            ),
            "maximum_representations_per_query": max(
                len(group) for group in source_groups.values()
            ),
        },
        "structure_extraction": {
            "version": 1,
            "join_graph_definition": (
                "source tables plus column-level cross-relation equality predicates, "
                "canonicalized independently of aliases"
            ),
            "join_topology_definition": (
                "source tables plus relation adjacency, ignoring aliases, columns, "
                "and filter literals"
            ),
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceb-dir", type=Path, default=DEFAULT_CEB_DIR)
    parser.add_argument(
        "--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.ceb_dir / "tasks.json"
    expected = build_inventory(args.ceb_dir, args.archive_manifest)
    if args.check:
        actual = json.loads(output.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("checked-in CEB task inventory is not reproducible")
        print("checked-in CEB task inventory verification passed")
        return
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing inventory: {output}")
    output.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "built CEB task inventory: "
        f"tasks={expected['task_count']} templates={expected['template_count']}"
    )


if __name__ == "__main__":
    main()
