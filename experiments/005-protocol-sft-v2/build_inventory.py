from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    DatasetSelection,
    ExcludedTask,
    JsonObject,
    SelectionCounts,
    SelectionMethod,
    SelectionSource,
    SelectionSplits,
    SelectionTask,
    require_list,
    require_string,
)
from qorl.util.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/ceb/tasks.json"
OUTPUT = ROOT / "experiments/005-protocol-sft-v2/selection.json"
SALT = "qorl-protocol-sft-v2"
SAMPLING_TASKS_PER_TEMPLATE = 25
LIVE_GATE_TASK_COUNT = 64
VALIDATION_TASK_COUNT = 64
VALIDATION_SEED = 20260830
EXCLUDED_TASKS = {
    "ceb-7a-7a117": "default PostgreSQL plan exceeded the calibration cap",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    template_id: str
    relation_count: int

    def selection_record(self) -> SelectionTask:
        return SelectionTask(task_id=self.task_id, template_id=self.template_id)


def salted_rank(kind: str, identifier: str) -> str:
    return hashlib.sha256(f"{SALT}:{kind}:{identifier}".encode()).hexdigest()


def load_tasks(source: JsonObject, partition: str) -> tuple[str, list[Task]]:
    inventory_id = source.get("inventory_id")
    records = source.get("tasks")
    if not isinstance(inventory_id, str) or not isinstance(records, list):
        raise ValueError("invalid CEB task inventory")

    tasks: list[Task] = []
    for record in records:
        if not isinstance(record, dict) or record.get("partition") != partition:
            continue
        task_id = record.get("task_id")
        template_id = record.get("template_id")
        relations = record.get("relations")
        if (
            not isinstance(task_id, str)
            or not isinstance(template_id, str)
            or not isinstance(relations, list)
        ):
            raise ValueError("invalid CEB task record")
        if task_id not in EXCLUDED_TASKS:
            tasks.append(Task(task_id, template_id, len(relations)))
    return inventory_id, tasks


def validation_rank(record: JsonObject) -> tuple[bool, str]:
    task_id = record.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("invalid CEB validation task")
    return (
        not bool(record.get("in_author_unique_plans_subset")),
        hashlib.sha256(f"{VALIDATION_SEED}:validation:{task_id}".encode()).hexdigest(),
    )


def select_frozen_validation(source: JsonObject) -> list[Task]:
    records = source.get("tasks")
    if not isinstance(records, list):
        raise ValueError("invalid CEB task inventory")
    by_template: dict[str, list[JsonObject]] = {}
    for value in records:
        if not isinstance(value, dict) or value.get("partition") != "validation":
            continue
        record = JSON_OBJECT_ADAPTER.validate_python(value)
        template_id = record.get("template_id")
        if not isinstance(template_id, str):
            raise ValueError("invalid CEB validation task")
        by_template.setdefault(template_id, []).append(record)

    selected: dict[str, list[Task]] = {}
    for template_id, values in sorted(by_template.items()):
        quota = VALIDATION_TASK_COUNT // len(by_template)
        ranked = sorted(values, key=validation_rank)
        selected[template_id] = [
            Task(
                task_id=require_string(record.get("task_id"), "task.task_id"),
                template_id=template_id,
                relation_count=len(
                    require_list(record.get("relations"), "task.relations")
                ),
            )
            for record in ranked[:quota]
        ]
    return interleave(selected, sorted(selected))


def quotas(templates: list[str], count: int) -> dict[str, int]:
    base, extra = divmod(count, len(templates))
    return {
        template_id: base + int(index < extra)
        for index, template_id in enumerate(templates)
    }


def interleave(
    selected: dict[str, list[Task]], template_order: list[str]
) -> list[Task]:
    return [
        selected[template_id][offset]
        for offset in range(max(map(len, selected.values())))
        for template_id in template_order
        if offset < len(selected[template_id])
    ]


def relation_counts(tasks: list[Task]) -> dict[str, int]:
    return {
        str(count): total
        for count, total in sorted(
            Counter(task.relation_count for task in tasks).items()
        )
    }


def build(source: JsonObject) -> DatasetSelection:
    inventory_id, tasks = load_tasks(source, "train")
    by_template: dict[str, list[Task]] = {}
    for task in tasks:
        by_template.setdefault(task.template_id, []).append(task)
    template_order = sorted(
        by_template,
        key=lambda template_id: salted_rank("template", template_id),
    )

    sampling: dict[str, list[Task]] = {}
    remaining: dict[str, list[Task]] = {}
    for template_id in template_order:
        ranked = sorted(
            by_template[template_id],
            key=lambda task: salted_rank("sampling", task.task_id),
        )
        if len(ranked) < SAMPLING_TASKS_PER_TEMPLATE:
            raise ValueError(f"{template_id} has too few sampling tasks")
        sampling[template_id] = ranked[:SAMPLING_TASKS_PER_TEMPLATE]
        remaining[template_id] = ranked[SAMPLING_TASKS_PER_TEMPLATE:]

    live_gate_quotas = quotas(template_order, LIVE_GATE_TASK_COUNT)
    live_gate: dict[str, list[Task]] = {}
    for template_id in template_order:
        ranked = sorted(
            remaining[template_id],
            key=lambda task: salted_rank("live-gate", task.task_id),
        )
        quota = live_gate_quotas[template_id]
        if len(ranked) < quota:
            raise ValueError(f"{template_id} has too few live-gate tasks")
        live_gate[template_id] = ranked[:quota]

    sampling_order = interleave(sampling, template_order)
    live_gate_order = interleave(live_gate, template_order)
    validation_order = select_frozen_validation(source)
    if set(sampling_order) & set(live_gate_order):
        raise RuntimeError("SFT sampling and live-gate selections overlap")

    sampling_records = [task.selection_record() for task in sampling_order]
    live_gate_records = [task.selection_record() for task in live_gate_order]
    validation_records = [task.selection_record() for task in validation_order]
    return DatasetSelection(
        inventory_id="qorl-protocol-sft-v2-selection-v1",
        selection=SelectionMethod(
            algorithm=(
                "lowest purpose-specific salted SHA-256 task IDs per template, "
                "emitted in salted template round-robin order"
            ),
            salt=SALT,
            template_order=template_order,
            sampling_tasks_per_template=SAMPLING_TASKS_PER_TEMPLATE,
            live_gate_template_quotas=live_gate_quotas,
            relation_count_note=(
                "relation count is fixed by template; exact template balance "
                "determines relation-count coverage"
            ),
            excluded_tasks=[
                ExcludedTask(task_id=task_id, reason=reason)
                for task_id, reason in sorted(EXCLUDED_TASKS.items())
            ],
            rl_v3_exclusion_splits=["sampling", "live_gate"],
        ),
        source=SelectionSource(
            inventory_id=inventory_id,
            path=str(SOURCE.relative_to(ROOT)),
            sha256=sha256_file(SOURCE),
        ),
        splits=SelectionSplits(
            sampling=sampling_records,
            live_gate=live_gate_records,
            validation=validation_records,
        ),
        counts=SelectionCounts(
            sampling=len(sampling_records),
            live_gate=len(live_gate_records),
            validation=len(validation_records),
            templates=len(template_order),
            sampling_by_relation_count=relation_counts(sampling_order),
            live_gate_by_relation_count=relation_counts(live_gate_order),
            validation_by_relation_count=relation_counts(validation_order),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.write:
        raise SystemExit("choose exactly one of --check or --write")

    source = JSON_OBJECT_ADAPTER.validate_json(SOURCE.read_text(encoding="utf-8"))
    rendered = json.dumps(build(source).to_wire(), indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale SFT v2 selection: {OUTPUT}")
        return
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
