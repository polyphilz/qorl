#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/ceb/tasks.json"
PILOT = ROOT / "experiments/003-rl-pilot-v1/selection.json"
OUTPUT = ROOT / "experiments/004-rl-run-v2/selection.json"
SALT = "qorl-rl-run-v2"
TASK_COUNT = 400
TASKS_PER_UPDATE = 4
ROLLOUTS_PER_TASK = 4
EXCLUDED_TASKS = {
    "ceb-7a-7a117": (
        "default PostgreSQL plan exceeded the 120-second calibration cap"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def salted_rank(kind: str, identifier: str) -> str:
    return hashlib.sha256(f"{SALT}:{kind}:{identifier}".encode()).hexdigest()


def build(source: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    prior_ids = {
        item["task_id"]
        for split in pilot["splits"].values()
        for item in split
    }
    prior_train_ids = {
        item["task_id"] for item in pilot["splits"]["train"]
    }
    by_template: dict[str, list[dict[str, Any]]] = {}
    for task in source["tasks"]:
        if (
            task["partition"] == "train"
            and task["task_id"] not in prior_ids
            and task["task_id"] not in EXCLUDED_TASKS
        ):
            by_template.setdefault(task["template_id"], []).append(task)

    template_order = sorted(
        by_template,
        key=lambda template_id: salted_rank("template", template_id),
    )
    base, extra = divmod(TASK_COUNT, len(template_order))
    quotas = {
        template_id: base + int(index < extra)
        for index, template_id in enumerate(template_order)
    }

    chosen: dict[str, list[dict[str, str]]] = {}
    for template_id in template_order:
        tasks = sorted(
            by_template[template_id],
            key=lambda task: salted_rank("task", task["task_id"]),
        )
        quota = quotas[template_id]
        if len(tasks) < quota:
            raise RuntimeError(f"{template_id} has fewer than {quota} unused tasks")
        chosen[template_id] = [
            {"task_id": task["task_id"], "template_id": template_id}
            for task in tasks[:quota]
        ]

    ordered = [
        chosen[template_id][round_index]
        for round_index in range(max(quotas.values()))
        for template_id in template_order
        if round_index < len(chosen[template_id])
    ]
    if len(ordered) != TASK_COUNT:
        raise RuntimeError(f"expected {TASK_COUNT} tasks, selected {len(ordered)}")
    batches = [
        ordered[index : index + TASKS_PER_UPDATE]
        for index in range(0, len(ordered), TASKS_PER_UPDATE)
    ]
    if any(len({item["template_id"] for item in batch}) != len(batch) for batch in batches):
        raise RuntimeError("an optimizer batch repeats a CEB template")

    return {
        "schema_version": 1,
        "inventory_id": "qorl-rl-run-v2",
        "selection": {
            "algorithm": (
                "lowest salted SHA-256 task IDs per template, emitted in salted "
                "template round-robin order"
            ),
            "salt": SALT,
            "template_order": template_order,
            "template_quotas": quotas,
            "training_allowed": ["train"],
            "excluded_tasks": [
                {"task_id": task_id, "reason": reason}
                for task_id, reason in sorted(EXCLUDED_TASKS.items())
            ],
            "prior_inventory": {
                "path": "experiments/003-rl-pilot-v1/selection.json",
                "sha256": sha256(PILOT),
                "excluded_distinct_task_count": len(prior_ids),
                "excluded_training_task_count": len(prior_train_ids),
            },
        },
        "run_shape": {
            "optimizer_updates": TASK_COUNT // TASKS_PER_UPDATE,
            "tasks_per_update": TASKS_PER_UPDATE,
            "rollouts_per_task": ROLLOUTS_PER_TASK,
            "effective_trajectory_count": TASK_COUNT * ROLLOUTS_PER_TASK,
        },
        "source": {
            "inventory_id": source["inventory_id"],
            "database": source["database"],
        },
        "splits": {"train": ordered},
        "counts": {
            "train": len(ordered),
            "templates": len(template_order),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.check and arguments.write:
        raise SystemExit("choose either --check or --write")

    expected = build(
        json.loads(SOURCE.read_text(encoding="utf-8")),
        json.loads(PILOT.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale RL run inventory: {OUTPUT}")
        return
    if arguments.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
        return
    print(rendered, end="")


if __name__ == "__main__":
    main()
