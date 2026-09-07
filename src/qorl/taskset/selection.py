"""Parse, sample, and validate topology-disjoint experiment task selections."""

import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

from qorl.taskset.exceptions import TaskSetError
from qorl.taskset.schemas import (
    BenchmarkId,
    SelectionExpression,
    TaskRole,
    TaskSelection,
    TemplateSample,
    TopologyOwner,
)
from qorl.taskset.taskset import TaskSet
from qorl.util.seeds import derive_seed

SELECTION_PATTERN = re.compile(
    r"(?P<role>[a-z]+)\s*=\s*(?P<benchmark>[a-z]+)"
    r"(?:\[(?P<templates>[^\[\]]+)\])?"
)
TEMPLATE_PATTERN = re.compile(r"(?P<template>[a-z0-9]+)\s*:\s*(?P<count>[0-9]+)")
SELECTION_SEED_PURPOSE = "task-selection"


def parse_selection(expression: str) -> SelectionExpression:
    """Parse a named whole benchmark or template:count list, with short template IDs."""
    match = SELECTION_PATTERN.fullmatch(expression.strip())
    if match is None:
        raise TaskSetError(f"invalid selection expression: {expression!r}")
    try:
        role = TaskRole(match["role"])
    except ValueError as error:
        raise TaskSetError(f"unknown selection role: {match['role']}") from error
    try:
        benchmark_id = BenchmarkId(match["benchmark"])
    except ValueError as error:
        raise TaskSetError(f"unknown benchmark: {match['benchmark']}") from error

    templates: list[TemplateSample] = []
    template_ids: set[str] = set()
    if match["templates"] is not None:
        for term in match["templates"].split(","):
            template = TEMPLATE_PATTERN.fullmatch(term.strip())
            if template is None:
                raise TaskSetError(f"invalid template selection: {term!r}")
            template_id = f"{benchmark_id.value}-{template['template']}"
            count = int(template["count"])
            if count < 1:
                raise TaskSetError(f"sample count must be positive for {template_id}")
            if template_id in template_ids:
                raise TaskSetError(f"duplicate template selection: {template_id}")
            template_ids.add(template_id)
            templates.append(TemplateSample(template_id, count))
    return SelectionExpression(role, benchmark_id, tuple(templates))


def select_tasks(
    expressions: Sequence[str],
    task_sets: Mapping[str, TaskSet],
    seed: int,
) -> dict[TaskRole, TaskSelection]:
    """Sample each template independently, then reject overlap between named roles."""
    if not expressions:
        raise TaskSetError("at least one selection expression is required")
    selections: dict[TaskRole, TaskSelection] = {}
    for expression in expressions:
        parsed = parse_selection(expression)
        if parsed.role in selections:
            raise TaskSetError(f"duplicate selection role: {parsed.role.value}")
        task_set = _benchmark(task_sets, parsed.benchmark_id)
        task_ids: list[str] = []
        if not parsed.templates:
            task_ids = [task.task_id for task in task_set.tasks]
        else:
            by_template: dict[str, list[str]] = defaultdict(list)
            for task in task_set.tasks:
                by_template[task.template_id].append(task.task_id)
            for sample in parsed.templates:
                metadata = task_set.tasks_metadata.get(sample.template_id)
                if metadata is None:
                    raise TaskSetError(f"unknown template: {sample.template_id}")
                if sample.count > metadata.query_count:
                    raise TaskSetError(
                        f"{sample.template_id}: requested {sample.count} queries, "
                        f"but only {metadata.query_count} are available"
                    )
                rng = random.Random(
                    derive_seed(
                        seed,
                        SELECTION_SEED_PURPOSE,
                        parsed.benchmark_id.value,
                        sample.template_id,
                    )
                )
                task_ids.extend(
                    rng.sample(sorted(by_template[sample.template_id]), sample.count)
                )
        selections[parsed.role] = TaskSelection(
            benchmark_id=parsed.benchmark_id, task_ids=sorted(task_ids)
        )
    validate_splits(selections, task_sets)
    return selections


def validate_splits(
    selections: Mapping[TaskRole, TaskSelection],
    task_sets: Mapping[str, TaskSet],
) -> None:
    """Reject shared query topologies across training, validation, and test splits.

    Topologies may repeat within a split. Compare stored catalog topology hashes,
    without reading SQL, for both newly sampled and imported selections.
    """
    owner_by_topology_hash: dict[str, TopologyOwner] = {}
    for role in TaskRole:
        selection = selections.get(role)
        if selection is None:
            continue
        task_set = _benchmark(task_sets, selection.benchmark_id)
        templates = sorted({task.template_id for task in task_set.resolve(selection)})
        for template_id in templates:
            topology_hash = task_set.tasks_metadata[template_id].topology_sha256
            previous = owner_by_topology_hash.get(topology_hash)
            if previous is not None and previous.role != role:
                raise TaskSetError(
                    f"topology overlap: {previous.role.value} {previous.template_id} "
                    f"and {role.value} {template_id}"
                )
            owner_by_topology_hash.setdefault(
                topology_hash, TopologyOwner(role=role, template_id=template_id)
            )


def _benchmark(task_sets: Mapping[str, TaskSet], benchmark_id: BenchmarkId) -> TaskSet:
    """Require the requested catalog, including for imported selections."""
    task_set = task_sets.get(benchmark_id.value)
    if task_set is None or task_set.task_set_id != benchmark_id.value:
        raise TaskSetError(f"benchmark catalog not loaded: {benchmark_id.value}")
    return task_set
