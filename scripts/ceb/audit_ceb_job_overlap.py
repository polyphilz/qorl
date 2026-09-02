#!/usr/bin/env python3
"""Compare CEB and JOB templates using alias-independent join fingerprints."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from qorl.util.hashing import sha256_bytes
from scripts.utils.query_structure import (
    extract_join_structure,
    task_join_fingerprints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_INVENTORY = REPOSITORY_ROOT / "data/job/tasks.json"
TEMPLATE_NAME = re.compile(r"^(?P<template>[0-9]+[a-z])", re.IGNORECASE)


def template_id(path: Path) -> str:
    for value in (path.parent.name, path.stem):
        match = TEMPLATE_NAME.match(value)
        if match:
            return f"ceb-{match['template'].lower()}"
    raise RuntimeError(f"cannot derive CEB template ID from {path}")


def read_ceb_sql(source_dir: Path, source_kind: str) -> list[dict[str, Any]]:
    suffix = ".sql" if source_kind == "sql" else ".toml"
    if source_kind == "sql":
        paths = sorted(source_dir.rglob(f"*{suffix}"))
    else:
        paths = sorted(
            path
            for path in source_dir.glob(f"*/*{suffix}")
            if TEMPLATE_NAME.fullmatch(path.parent.name)
        )
    if not paths:
        raise RuntimeError(f"no {suffix} files found under {source_dir}")

    queries: list[dict[str, Any]] = []
    for path in paths:
        content = path.read_bytes()
        if source_kind == "sql":
            sql = content.decode("utf-8")
        else:
            document = tomllib.loads(content.decode("utf-8"))
            try:
                sql = document["base_sql"]["sql"]
            except KeyError:
                continue
        structure = extract_join_structure(sql, str(path))
        graph_hash, topology_hash = task_join_fingerprints(structure)
        if graph_hash != structure["join_graph_sha256"]:
            raise RuntimeError(f"join-graph fingerprint disagreement for {path}")
        queries.append(
            {
                "template_id": template_id(path),
                "source_name": path.name,
                "source_sha256": sha256_bytes(content),
                "relation_count": structure["relation_count"],
                "join_predicate_count": structure["join_predicate_count"],
                "join_graph_sha256": graph_hash,
                "join_topology_sha256": topology_hash,
            }
        )
    if not queries:
        raise RuntimeError(f"no usable CEB queries found under {source_dir}")
    return queries


def job_fingerprints(inventory: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    graphs: dict[str, set[str]] = defaultdict(set)
    topologies: dict[str, set[str]] = defaultdict(set)
    for task in inventory["tasks"]:
        graph_hash, topology_hash = task_join_fingerprints(task)
        if graph_hash != task["join_graph_sha256"]:
            raise RuntimeError(
                f"checked-in JOB fingerprint disagreement for {task['task_id']}"
            )
        graphs[graph_hash].add(task["template_id"])
        topologies[topology_hash].add(task["template_id"])
    return graphs, topologies


def build_report(
    queries: list[dict[str, Any]],
    job_inventory: dict[str, Any],
    source_kind: str,
    source_repository: str | None,
    source_commit: str | None,
) -> dict[str, Any]:
    job_graphs, job_topologies = job_fingerprints(job_inventory)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_template[query["template_id"]].append(query)

    templates: list[dict[str, Any]] = []
    excluded = 0
    for ceb_template, members in sorted(by_template.items()):
        graph_hashes = sorted({member["join_graph_sha256"] for member in members})
        topology_hashes = sorted(
            {member["join_topology_sha256"] for member in members}
        )
        exact_matches = sorted(
            {
                job_template
                for graph_hash in graph_hashes
                for job_template in job_graphs.get(graph_hash, set())
            }
        )
        topology_matches = sorted(
            {
                job_template
                for topology_hash in topology_hashes
                for job_template in job_topologies.get(topology_hash, set())
            }
        )
        disposition = "excluded" if topology_matches else "eligible"
        excluded += disposition == "excluded"
        templates.append(
            {
                "template_id": ceb_template,
                "query_count": len(members),
                "relation_counts": sorted(
                    {member["relation_count"] for member in members}
                ),
                "join_graph_sha256": graph_hashes,
                "join_topology_sha256": topology_hashes,
                "exact_job_template_matches": exact_matches,
                "topology_job_template_matches": topology_matches,
                "disposition": disposition,
            }
        )

    return {
        "schema_version": 1,
        "comparison": {
            "rule": (
                "exclude a complete CEB template when any of its alias-independent "
                "table-colored join topologies matches a JOB template"
            ),
            "exact_graph_definition": (
                "source tables plus column-level cross-relation equality predicates"
            ),
            "topology_definition": (
                "source tables plus relation adjacency, ignoring aliases, columns, "
                "and filter literals"
            ),
        },
        "ceb_source": {
            "kind": source_kind,
            **({"repository_url": source_repository} if source_repository else {}),
            **({"commit": source_commit} if source_commit else {}),
        },
        "job_inventory": {
            "inventory_id": job_inventory["inventory_id"],
            "task_count": job_inventory["task_count"],
            "template_count": job_inventory["template_count"],
        },
        "summary": {
            "ceb_query_count": len(queries),
            "ceb_template_count": len(templates),
            "eligible_template_count": len(templates) - excluded,
            "excluded_template_count": excluded,
        },
        "templates": templates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceb-source-dir", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("sql", "toml"), default="sql")
    parser.add_argument("--source-repository")
    parser.add_argument("--source-commit")
    parser.add_argument("--job-inventory", type=Path, default=DEFAULT_JOB_INVENTORY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    job_inventory = json.loads(args.job_inventory.read_text(encoding="utf-8"))
    report = build_report(
        read_ceb_sql(args.ceb_source_dir, args.source_kind),
        job_inventory,
        args.source_kind,
        args.source_repository,
        args.source_commit,
    )
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("checked-in CEB/JOB overlap report is not reproducible")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    summary = report["summary"]
    prefix = (
        "CEB/JOB overlap audit verified: "
        if args.check
        else "CEB/JOB overlap audit complete: "
    )
    print(
        f"{prefix}templates={summary['ceb_template_count']} "
        f"eligible={summary['eligible_template_count']} "
        f"excluded={summary['excluded_template_count']}"
    )


if __name__ == "__main__":
    main()
