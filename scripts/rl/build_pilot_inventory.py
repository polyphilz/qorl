#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/ceb/ceb-v1/tasks.json"
OUTPUT = ROOT / "data/ceb/ceb-v1/rl-pilot-v1.json"
SALT = "qorl-rl-pilot-v1"


def rank(task: dict[str, Any], partition: str) -> str:
    return hashlib.sha256(
        f"{SALT}:{partition}:{task['task_id']}".encode()
    ).hexdigest()


def build(source: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, list[dict[str, str]]] = {}
    counts = {"train": 4, "validation": 4}
    for partition, count in counts.items():
        by_template: dict[str, list[dict[str, Any]]] = {}
        for task in source["tasks"]:
            if task["partition"] == partition:
                by_template.setdefault(task["template_id"], []).append(task)
        chosen = []
        for template_id, tasks in sorted(by_template.items()):
            tasks.sort(key=lambda task: rank(task, partition))
            if len(tasks) < count:
                raise RuntimeError(f"{template_id} has fewer than {count} tasks")
            chosen.extend(
                {
                    "task_id": task["task_id"],
                    "template_id": template_id,
                }
                for task in tasks[:count]
            )
        selected[partition] = chosen
    selected["spike"] = selected["train"][:1]
    return {
        "schema_version": 1,
        "inventory_id": "qorl-rl-pilot-v1",
        "selection": {
            "algorithm": "four lowest salted SHA-256 task IDs per template",
            "salt": SALT,
            "training_allowed": ["spike", "train"],
            "checkpoint_selection_allowed": ["validation"],
        },
        "source": {
            "inventory_id": source["inventory_id"],
            "database": source["database"],
        },
        "splits": selected,
        "counts": {name: len(tasks) for name, tasks in selected.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = build(json.loads(SOURCE.read_text(encoding="utf-8")))
    rendered = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale RL pilot inventory: {OUTPUT}")
        return
    print(rendered, end="")


if __name__ == "__main__":
    main()
