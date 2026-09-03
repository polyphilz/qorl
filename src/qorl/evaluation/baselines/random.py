from __future__ import annotations

import random
from enum import StrEnum
from typing import Any

from qorl.plans.catalog import TaskCatalog
from qorl.plans.exceptions import ActionError
from qorl.plans.random_tree import random_join_tree
from qorl.plans.schemas import (
    ACTION_SCHEMA_VERSION,
    AUTO,
    MAX_PARALLEL_WORKERS,
    JoinMethod,
    MemoizeMode,
    ParallelMode,
    PlanAction,
    RowMode,
    ScanMethod,
)

SAMPLER_VERSION = 2


class ActionFamily(StrEnum):
    LEADING = "leading"
    JOIN = "join"
    SCAN = "scan"
    ROWS = "rows"
    PARALLEL = "parallel"
    SETTING = "setting"
    DISABLED_INDEX = "disabled_index"


class JoinDirective(StrEnum):
    FORCE = "force"
    FORBID = "forbid"
    MEMOIZE = "memoize"


MIN_FAMILIES_PER_ACTION = 1
MAX_FAMILIES_PER_ACTION = 3
MAX_SAMPLE_ATTEMPTS = 100
JOIN_DIRECTIVE_WEIGHTS = {
    JoinDirective.FORCE.value: 70,
    JoinDirective.FORBID.value: 20,
    JoinDirective.MEMOIZE.value: 10,
}
SCAN_FORCE_PROBABILITY = 0.8
NAMED_INDEX_PROBABILITY = 0.5
ROW_MULTIPLIER_VALUES = (0.1, 0.25, 0.5, 2, 4, 10)
ROW_ABSOLUTE_VALUES = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)

FAMILY_WEIGHTS = {
    ActionFamily.LEADING.value: 30,
    ActionFamily.JOIN.value: 25,
    ActionFamily.SCAN.value: 25,
    ActionFamily.ROWS.value: 10,
    ActionFamily.PARALLEL.value: 4,
    ActionFamily.SETTING.value: 4,
    ActionFamily.DISABLED_INDEX.value: 2,
}

SETTING_VALUES: dict[str, list[bool | int | float]] = {
    "enable_hashjoin": [False],
    "enable_mergejoin": [False],
    "enable_nestloop": [False],
    "enable_seqscan": [False],
    "enable_indexscan": [False],
    "enable_bitmapscan": [False],
    "enable_memoize": [False],
    "random_page_cost": [1.0, 2.0, 8.0],
    "seq_page_cost": [0.5, 2.0],
    "cpu_tuple_cost": [0.005, 0.02],
    "join_collapse_limit": [1, 16],
}


def weighted_families(rng: random.Random, count: int) -> list[str]:
    available = list(FAMILY_WEIGHTS)
    selected: list[str] = []
    for _ in range(count):
        weights = [FAMILY_WEIGHTS[name] for name in available]
        name = rng.choices(available, weights=weights, k=1)[0]
        available.remove(name)
        selected.append(name)
    return selected


def join_targets(tree: str | dict[str, Any]) -> list[list[str]]:
    targets: list[list[str]] = []

    def visit(node: str | dict[str, Any]) -> set[str]:
        if isinstance(node, str):
            return {node}
        relations = visit(node["left"]) | visit(node["right"])
        targets.append(sorted(relations))
        return relations

    visit(tree)
    return targets


def add_join(
    action: dict[str, Any], targets: list[list[str]], rng: random.Random
) -> None:
    item: dict[str, Any] = {
        "relations": rng.choice(targets),
        "force": AUTO,
        "forbid": [],
        "memoize": MemoizeMode.AUTO.value,
    }
    directives = list(JOIN_DIRECTIVE_WEIGHTS)
    directive = rng.choices(
        directives,
        weights=[JOIN_DIRECTIVE_WEIGHTS[name] for name in directives],
        k=1,
    )[0]
    if directive == JoinDirective.FORCE:
        item["force"] = rng.choice([method.value for method in JoinMethod])
    elif directive == JoinDirective.FORBID:
        item["forbid"] = [rng.choice([method.value for method in JoinMethod])]
    else:
        item["memoize"] = rng.choice(
            [MemoizeMode.FORCE.value, MemoizeMode.FORBID.value]
        )
    action["joins"] = [item]


def add_scan(action: dict[str, Any], catalog: TaskCatalog, rng: random.Random) -> None:
    relation = rng.choice(sorted(catalog.relations))
    indexes = sorted(catalog.indexes.get(relation, []))
    methods = [ScanMethod.SEQ.value]
    if indexes:
        methods.extend(
            [
                ScanMethod.INDEX.value,
                ScanMethod.INDEX_ONLY.value,
                ScanMethod.BITMAP.value,
            ]
        )
    force = rng.random() < SCAN_FORCE_PROBABILITY
    method = rng.choice(methods)
    item: dict[str, Any] = {
        "relation": relation,
        "force": method if force else AUTO,
        "forbid": [] if force else [method],
        "indexes": [],
    }
    if (
        force
        and method in {ScanMethod.INDEX, ScanMethod.INDEX_ONLY, ScanMethod.BITMAP}
        and rng.random() < NAMED_INDEX_PROBABILITY
    ):
        item["indexes"] = [rng.choice(indexes)]
    action["scans"] = [item]


def add_rows(
    action: dict[str, Any], targets: list[list[str]], rng: random.Random
) -> None:
    mode = rng.choice([mode.value for mode in RowMode])
    values = ROW_MULTIPLIER_VALUES if mode == RowMode.MULTIPLY else ROW_ABSOLUTE_VALUES
    action["row_corrections"] = [
        {"relations": rng.choice(targets), "mode": mode, "value": rng.choice(values)}
    ]


def add_parallel(
    action: dict[str, Any], catalog: TaskCatalog, rng: random.Random
) -> None:
    action["parallel"] = [
        {
            "relation": rng.choice(sorted(catalog.relations)),
            "workers": rng.choice(range(MAX_PARALLEL_WORKERS + 1)),
            "mode": rng.choice([mode.value for mode in ParallelMode]),
        }
    ]


def add_setting(action: dict[str, Any], rng: random.Random) -> None:
    name = rng.choice(sorted(SETTING_VALUES))
    action["settings"] = {name: rng.choice(SETTING_VALUES[name])}


def add_disabled_index(
    action: dict[str, Any], catalog: TaskCatalog, rng: random.Random
) -> None:
    relations = sorted(
        relation for relation, indexes in catalog.indexes.items() if indexes
    )
    if not relations:
        return
    relation = rng.choice(relations)
    action["disabled_indexes"] = [
        {
            "relation": relation,
            "indexes": [rng.choice(sorted(catalog.indexes[relation]))],
        }
    ]


def sample_action(catalog: TaskCatalog, rng: random.Random) -> dict[str, Any]:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        tree = random_join_tree(catalog, rng)
        targets = join_targets(tree)
        action: dict[str, Any] = {"version": ACTION_SCHEMA_VERSION}
        for family in weighted_families(
            rng, rng.randint(MIN_FAMILIES_PER_ACTION, MAX_FAMILIES_PER_ACTION)
        ):
            if family == ActionFamily.LEADING:
                action["leading"] = tree
            elif family == ActionFamily.JOIN:
                add_join(action, targets, rng)
            elif family == ActionFamily.SCAN:
                add_scan(action, catalog, rng)
            elif family == ActionFamily.ROWS:
                add_rows(action, targets, rng)
            elif family == ActionFamily.PARALLEL:
                add_parallel(action, catalog, rng)
            elif family == ActionFamily.SETTING:
                add_setting(action, rng)
            elif family == ActionFamily.DISABLED_INDEX:
                add_disabled_index(action, catalog, rng)
        try:
            return PlanAction.from_raw(action, catalog).to_wire()
        except ActionError:
            continue
    raise RuntimeError("could not sample a schema-valid PlanAction")


def sampler_manifest() -> dict[str, Any]:
    return {
        "version": SAMPLER_VERSION,
        "families_per_action": {
            "minimum": MIN_FAMILIES_PER_ACTION,
            "maximum": MAX_FAMILIES_PER_ACTION,
            "distribution": "uniform",
        },
        "family_weights": FAMILY_WEIGHTS,
        "join_directive_weights": JOIN_DIRECTIVE_WEIGHTS,
        "scan_force_probability": SCAN_FORCE_PROBABILITY,
        "named_index_probability_given_index_scan": NAMED_INDEX_PROBABILITY,
        "tid_scan_sampling": False,
        "disabled_index_sampling": True,
        "setting_values": SETTING_VALUES,
        "postgresql_applicability_filtering": False,
    }
