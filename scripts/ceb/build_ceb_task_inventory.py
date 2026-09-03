#!/usr/bin/env python3
"""Build or verify the post-leakage-audit CEB task inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qorl.util.hashing import sha256_bytes, sha256_file
from qorl.workload.ceb import SPLIT_SALT, choose_validation_templates
from qorl.workload.query_structure import (
    extract_join_structure,
    task_join_fingerprints,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CEB_DIR = REPOSITORY_ROOT / "data/ceb"
DEFAULT_JOB_INVENTORY = REPOSITORY_ROOT / "data/job/tasks.json"


def build_inventory(ceb_dir: Path, job_inventory_path: Path) -> dict[str, Any]:
    provenance_dir = ceb_dir / "provenance"
    sources_path = provenance_dir / "sources.json"
    unique_path = provenance_dir / "unique-plans.json"
    overlap_path = provenance_dir / "job-overlap.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    unique = json.loads(unique_path.read_text(encoding="utf-8"))
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    job_inventory = json.loads(job_inventory_path.read_text(encoding="utf-8"))

    dispositions = {
        template["template_id"]: template["disposition"]
        for template in overlap["templates"]
    }
    source_templates = set(sources["template_query_counts"])
    if set(dispositions) != source_templates:
        raise RuntimeError("overlap report and recovered source templates differ")
    eligible = {
        template
        for template, disposition in dispositions.items()
        if disposition == "eligible"
    }
    excluded = source_templates - eligible
    source_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sql_templates: dict[str, set[str]] = defaultdict(set)
    for source in sources["queries"]:
        if source["template_id"] in eligible:
            source_groups[(source["template_id"], source["sql_sha256"])].append(source)
            sql_templates[source["sql_sha256"]].add(source["template_id"])
    cross_template_duplicates = {
        sql_sha256: templates
        for sql_sha256, templates in sql_templates.items()
        if len(templates) > 1
    }
    if cross_template_duplicates:
        raise RuntimeError("exact SQL occurs in multiple split templates")
    counts = Counter(template for template, _sql_sha256 in source_groups)
    validation_templates = set(choose_validation_templates(counts))
    train_templates = eligible - validation_templates
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
            "partition": (
                "validation"
                if source["template_id"] in validation_templates
                else "train"
            ),
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

    partition_counts = Counter(task["partition"] for task in tasks)
    partition_templates = {
        "train": sorted(train_templates),
        "validation": sorted(validation_templates),
    }
    if len(tasks) != sum(counts.values()):
        raise RuntimeError("eligible query count differs from built task count")
    return {
        "schema_version": 1,
        "inventory_id": "ceb-v1-tasks-v1",
        "role": "training_and_validation",
        "task_count": len(tasks),
        "template_count": len(eligible),
        "database": job_inventory["database"],
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
        "job_leakage_audit": {
            "report_path": "provenance/job-overlap.json",
            "report_sha256": sha256_file(overlap_path),
            "rule": overlap["comparison"]["rule"],
            "source_template_count": len(source_templates),
            "eligible_template_count": len(eligible),
            "excluded_template_count": len(excluded),
            "excluded_templates": sorted(excluded),
        },
        "split": {
            "unit": "template",
            "target_validation_fraction": 0.25,
            "algorithm": (
                "hold out ceil(25% of eligible templates), choosing the set whose "
                "query count is nearest 25% of eligible queries; break ties by "
                "SHA-256"
            ),
            "salt": SPLIT_SALT,
            "partitions": {
                "train": {
                    "training_allowed": True,
                    "checkpoint_selection_allowed": False,
                    "template_ids": partition_templates["train"],
                    "template_count": len(train_templates),
                    "task_count": partition_counts["train"],
                },
                "validation": {
                    "training_allowed": False,
                    "checkpoint_selection_allowed": True,
                    "template_ids": partition_templates["validation"],
                    "template_count": len(validation_templates),
                    "task_count": partition_counts["validation"],
                },
            },
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
    parser.add_argument("--job-inventory", type=Path, default=DEFAULT_JOB_INVENTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.ceb_dir / "tasks.json"
    expected = build_inventory(args.ceb_dir, args.job_inventory)
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
        f"tasks={expected['task_count']} templates={expected['template_count']} "
        f"train={expected['split']['partitions']['train']['task_count']} "
        f"validation={expected['split']['partitions']['validation']['task_count']}"
    )


if __name__ == "__main__":
    main()
