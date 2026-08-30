from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qorl.fixture import TaskSet, sha256_file
from scripts.utils.protocol_demo import validate_protocol_demo


DATASET_ID = "protocol-sft-v1"
DATASET_SEED = 20260830
SPLIT_COUNTS = {"train": 256, "validation": 64}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def ranked_tasks(
    tasks: list[dict[str, Any]], partition: str, seed: int
) -> dict[str, list[dict[str, Any]]]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["partition"] == partition:
            by_template[task["template_id"]].append(task)
    return {
        template: sorted(
            values,
            key=lambda task: (
                not task["in_author_unique_plans_subset"],
                hashlib.sha256(
                    f"{seed}:{partition}:{task['task_id']}".encode()
                ).hexdigest(),
            ),
        )
        for template, values in sorted(by_template.items())
    }


def template_quotas(
    ranked: dict[str, list[dict[str, Any]]], count: int
) -> dict[str, int]:
    templates = sorted(ranked)
    if not templates:
        raise ValueError("no CEB templates")
    base, remainder = divmod(count, len(templates))
    return {
        template: base + (index < remainder)
        for index, template in enumerate(templates)
    }


def select_tasks(
    tasks: list[dict[str, Any]], partition: str, count: int, seed: int
) -> list[dict[str, Any]]:
    """Choose a template-balanced, stable set, preferring CEB's unique plans."""
    ranked = ranked_tasks(tasks, partition, seed)
    quotas = template_quotas(ranked, count)

    selected: dict[str, list[dict[str, Any]]] = {}
    for template, quota in quotas.items():
        if len(ranked[template]) < quota:
            raise ValueError(f"{template} cannot supply {quota} {partition} tasks")
        selected[template] = ranked[template][:quota]

    # Interleave templates so a partial/resumed run remains representative.
    return [
        selected[template][offset]
        for offset in range(max(map(len, selected.values())))
        for template in sorted(selected)
        if offset < len(selected[template])
    ]


def action_families(action: dict[str, Any]) -> list[str]:
    families: set[str] = set()
    if "leading" in action:
        families.add("leading")
    for join in action.get("joins", []):
        if join.get("force") not in {None, "auto"} or join.get("forbid"):
            families.add("join")
        if join.get("memoize") not in {None, "auto"}:
            families.add("memoize")
    if action.get("scans"):
        families.add("scan")
        if any(scan.get("indexes") for scan in action["scans"]):
            families.add("index_selection")
    if action.get("disabled_indexes"):
        families.add("index_exclusion")
    if action.get("row_corrections"):
        families.add("rows")
    if action.get("parallel"):
        families.add("parallel")
    if action.get("settings"):
        families.add("setting")
    return sorted(families)


def distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    p90 = ordered[math.ceil(0.9 * len(ordered)) - 1]
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": p90,
        "maximum": ordered[-1],
    }


def speedup_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "status": "not_measured",
            "selection_used_speed": False,
        }
    bins = Counter()
    for value in values:
        if value < 0.5:
            bins["<0.5"] += 1
        elif value < 0.9:
            bins["0.5-0.9"] += 1
        elif value <= 1.1:
            bins["0.9-1.1"] += 1
        elif value <= 2.0:
            bins["1.1-2.0"] += 1
        else:
            bins[">2.0"] += 1
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "bins": dict(sorted(bins.items())),
        "selection_used_speed": False,
    }


def load_documents(output_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for partition in SPLIT_COUNTS:
        directory = output_dir / "demonstrations" / partition
        for path in directory.glob("*.json"):
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return sorted(records, key=lambda item: item[1]["metadata"]["ordinal"])


def finalize_dataset(
    repository: Path,
    output_dir: Path,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    task_set = TaskSet.load(repository, "ceb-v1")
    overlap_path = repository / "data/ceb/ceb-v1/job-overlap.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    eligible = {
        item["template_id"]
        for item in overlap["templates"]
        if item["disposition"] == "eligible"
    }

    records = load_documents(output_dir)
    counts = Counter(document["metadata"]["partition"] for _, document in records)
    if dict(counts) != SPLIT_COUNTS:
        raise ValueError(f"demonstration counts differ: {dict(counts)}")

    tool_calls = Counter()
    candidate_counts = Counter()
    recipes = Counter()
    measurement_modes = Counter()
    family_counts = Counter()
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    turn_counts: list[int] = []
    action_hashes: set[str] = set()
    plan_hashes: set[str] = set()
    speedups: list[float] = []
    demonstrations: list[dict[str, Any]] = []
    split_task_ids: dict[str, set[str]] = defaultdict(set)
    split_templates: dict[str, set[str]] = defaultdict(set)

    prime_rows: dict[str, list[bytes]] = defaultdict(list)
    for path, document in records:
        validation = validate_protocol_demo(document, repository)
        metadata = document["metadata"]
        partition = metadata["partition"]
        template = metadata["template_id"]
        if template not in eligible:
            raise ValueError(f"ineligible CEB template in dataset: {template}")
        split_task_ids[partition].add(metadata["task_id"])
        split_templates[partition].add(template)
        template_counts[partition][template] += 1
        candidate_counts[str(len(validation["candidate_ids"]))] += 1
        turn_counts.append(validation["turn_count"])
        recipes[metadata["inspection_recipe"]] += 1
        measurement_modes[metadata.get("measurement_mode", "measured")] += 1

        for message in document["messages"]:
            if message["role"] == "assistant":
                name = message["tool_calls"][0]["function"]["name"]
                tool_calls[name] += 1
                if name == "evaluate_candidate":
                    arguments = json.loads(
                        message["tool_calls"][0]["function"]["arguments"]
                    )
                    action = arguments["action"]
                    action_hashes.add(canonical_sha256(action))
                    family_counts.update(action_families(action))
            elif message["role"] == "tool" and message["name"] == "evaluate_candidate":
                result = json.loads(message["content"])
                plan_hashes.add(result["plan_sha256"])
                speedup = result.get("provisional_speedup")
                if isinstance(speedup, (int, float)):
                    speedups.append(float(speedup))

        relative = path.relative_to(output_dir)
        demonstrations.append(
            {
                "demonstration_id": metadata["demonstration_id"],
                "partition": partition,
                "task_id": metadata["task_id"],
                "template_id": template,
                "path": relative.as_posix(),
                "canonical_sha256": validation["canonical_sha256"],
            }
        )
        row = {
            "messages": document["messages"],
            "tools": json.dumps(document["tools"], sort_keys=True),
        }
        prime_rows[partition].append((canonical_json(row) + "\n").encode())

    if split_task_ids["train"] & split_task_ids["validation"]:
        raise ValueError("train and validation contain the same CEB task")
    if split_templates["train"] & split_templates["validation"]:
        raise ValueError("train and validation templates overlap")

    prime_dir = output_dir / "prime"
    prime_dir.mkdir(parents=True, exist_ok=True)
    prime_artifacts: dict[str, dict[str, Any]] = {}
    for partition, expected in SPLIT_COUNTS.items():
        rows = prime_rows[partition]
        if len(rows) != expected:
            raise ValueError(f"Prime {partition} row count differs")
        encoded = b"".join(rows)
        path = prime_dir / f"{partition}.jsonl"
        path.write_bytes(encoded)
        prime_artifacts[partition] = {
            "path": path.relative_to(output_dir).as_posix(),
            "rows": len(rows),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "seed": DATASET_SEED,
        "task_set_id": task_set.task_set_id,
        "database": task_set.inventory["database"],
        "sources": {
            "ceb_inventory": {
                "path": task_set.inventory_path.relative_to(repository).as_posix(),
                "sha256": sha256_file(task_set.inventory_path),
            },
            "job_overlap_audit": {
                "path": overlap_path.relative_to(repository).as_posix(),
                "sha256": sha256_file(overlap_path),
                "eligible_templates": len(eligible),
                "excluded_templates": overlap["summary"]["excluded_template_count"],
            },
        },
        "counts": dict(sorted(counts.items())),
        "prime_artifacts": prime_artifacts,
        "leakage": {
            "job_tasks": 0,
            "ineligible_ceb_templates": 0,
            "shared_tasks_between_splits": 0,
            "shared_templates_between_splits": 0,
        },
        "statistics": {
            "templates": {
                partition: dict(sorted(values.items()))
                for partition, values in sorted(template_counts.items())
            },
            "author_unique_plan_tasks": Counter(
                document["metadata"]["partition"]
                for _, document in records
                if document["metadata"]["in_author_unique_plans_subset"]
            ),
            "inspection_recipes": dict(sorted(recipes.items())),
            "measurement_modes": dict(sorted(measurement_modes.items())),
            "tool_calls": dict(sorted(tool_calls.items())),
            "candidate_counts": dict(sorted(candidate_counts.items())),
            "turn_counts": distribution(turn_counts),
            "action_families": dict(sorted(family_counts.items())),
            "candidate_attempts": tool_calls["evaluate_candidate"],
            "unique_normalized_actions": len(action_hashes),
            "unique_physical_plans": len(plan_hashes),
            "provisional_speedups": speedup_distribution(speedups),
        },
        "generation_failures": failures,
        "demonstrations": demonstrations,
    }
    # Counter is JSON-compatible, but normalize it for a stable manifest.
    manifest["statistics"]["author_unique_plan_tasks"] = dict(
        sorted(manifest["statistics"]["author_unique_plan_tasks"].items())
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_dataset(repository: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected protocol dataset ID")
    records = load_documents(output_dir)
    by_path = {
        item["path"]: item for item in manifest.get("demonstrations", [])
    }
    counts = Counter()
    for path, document in records:
        relative = path.relative_to(output_dir).as_posix()
        expected = by_path.get(relative)
        if expected is None:
            raise ValueError(f"unmanifested demonstration: {relative}")
        validation = validate_protocol_demo(document, repository)
        if validation["canonical_sha256"] != expected["canonical_sha256"]:
            raise ValueError(f"demonstration checksum mismatch: {relative}")
        counts[document["metadata"]["partition"]] += 1
    if dict(counts) != SPLIT_COUNTS:
        raise ValueError(f"demonstration counts differ: {dict(counts)}")
    if len(records) != len(by_path):
        raise ValueError("manifest demonstration count differs")

    for partition, artifact in manifest["prime_artifacts"].items():
        path = output_dir / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Prime {partition} checksum mismatch")
        if len(path.read_bytes().splitlines()) != artifact["rows"]:
            raise ValueError(f"Prime {partition} row count differs")
    return {
        "dataset_id": DATASET_ID,
        "demonstrations": len(records),
        "counts": dict(sorted(counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
    }
