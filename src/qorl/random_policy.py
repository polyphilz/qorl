from __future__ import annotations

import random
from typing import Any

from qorl.action import ActionError, TaskCatalog, normalize_action


SAMPLER_VERSION = 1
FAMILY_WEIGHTS = {
    "leading": 30,
    "join": 25,
    "scan": 25,
    "rows": 10,
    "parallel": 4,
    "setting": 4,
    "disabled_index": 2,
}

SETTING_VALUES: dict[str, list[bool | int | float]] = {
    "enable_hashjoin": [False],
    "enable_mergejoin": [False],
    "enable_nestloop": [False],
    "enable_seqscan": [False],
    "enable_indexscan": [False],
    "enable_bitmapscan": [False],
    "enable_memoize": [False],
    "geqo": [False],
    "random_page_cost": [1.0, 2.0, 8.0],
    "seq_page_cost": [0.5, 2.0],
    "cpu_tuple_cost": [0.005, 0.02],
    "join_collapse_limit": [1, 16],
    "geqo_threshold": [8, 16],
    "max_parallel_workers_per_gather": [0, 1, 4, 8],
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


def random_join_tree(
    catalog: TaskCatalog, rng: random.Random
) -> str | dict[str, Any]:
    components: list[tuple[set[str], str | dict[str, Any]]] = [
        ({relation}, relation) for relation in sorted(catalog.relations)
    ]
    while len(components) > 1:
        pairs = [
            (left, right)
            for left in range(len(components))
            for right in range(left + 1, len(components))
            if any(
                catalog.adjacency[relation] & components[right][0]
                for relation in components[left][0]
            )
        ]
        left_index, right_index = rng.choice(pairs)
        left_relations, left_tree = components[left_index]
        right_relations, right_tree = components[right_index]
        if rng.random() < 0.5:
            left_tree, right_tree = right_tree, left_tree
        merged = (
            left_relations | right_relations,
            {"left": left_tree, "right": right_tree},
        )
        components.pop(right_index)
        components.pop(left_index)
        components.append(merged)
    return components[0][1]


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


def add_join(action: dict[str, Any], targets: list[list[str]], rng: random.Random) -> None:
    item: dict[str, Any] = {
        "relations": rng.choice(targets),
        "force": "auto",
        "forbid": [],
        "memoize": "auto",
    }
    directive = rng.choices(
        ["force", "forbid", "memoize"], weights=[70, 20, 10], k=1
    )[0]
    if directive == "force":
        item["force"] = rng.choice(["hash", "merge", "nestloop"])
    elif directive == "forbid":
        item["forbid"] = [rng.choice(["hash", "merge", "nestloop"])]
    else:
        item["memoize"] = rng.choice(["force", "forbid"])
    action["joins"] = [item]


def add_scan(action: dict[str, Any], catalog: TaskCatalog, rng: random.Random) -> None:
    relation = rng.choice(sorted(catalog.relations))
    indexes = sorted(catalog.indexes.get(relation, []))
    methods = ["seq"]
    if indexes:
        methods.extend(["index", "index_only", "bitmap"])
    force = rng.random() < 0.8
    method = rng.choice(methods)
    item: dict[str, Any] = {
        "relation": relation,
        "force": method if force else "auto",
        "forbid": [] if force else [method],
        "indexes": [],
    }
    if (
        force
        and method in {"index", "index_only", "bitmap"}
        and rng.random() < 0.5
    ):
        item["indexes"] = [rng.choice(indexes)]
    action["scans"] = [item]


def add_rows(
    action: dict[str, Any], targets: list[list[str]], rng: random.Random
) -> None:
    mode = rng.choice(["absolute", "add", "subtract", "multiply"])
    values = (
        [0.1, 0.25, 0.5, 2, 4, 10]
        if mode == "multiply"
        else [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    )
    action["row_corrections"] = [
        {"relations": rng.choice(targets), "mode": mode, "value": rng.choice(values)}
    ]


def add_parallel(
    action: dict[str, Any], catalog: TaskCatalog, rng: random.Random
) -> None:
    action["parallel"] = [
        {
            "relation": rng.choice(sorted(catalog.relations)),
            "workers": rng.choice([0, 1, 2, 4, 8]),
            "mode": rng.choice(["soft", "hard"]),
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
    for _ in range(100):
        tree = random_join_tree(catalog, rng)
        targets = join_targets(tree)
        action: dict[str, Any] = {"version": 1}
        for family in weighted_families(rng, rng.randint(1, 3)):
            if family == "leading":
                action["leading"] = tree
            elif family == "join":
                add_join(action, targets, rng)
            elif family == "scan":
                add_scan(action, catalog, rng)
            elif family == "rows":
                add_rows(action, targets, rng)
            elif family == "parallel":
                add_parallel(action, catalog, rng)
            elif family == "setting":
                add_setting(action, rng)
            elif family == "disabled_index":
                add_disabled_index(action, catalog, rng)
        try:
            return normalize_action(action, catalog)
        except ActionError:
            continue
    raise RuntimeError("could not sample a schema-valid PlanAction")


def sampler_manifest() -> dict[str, Any]:
    return {
        "version": SAMPLER_VERSION,
        "families_per_action": {
            "minimum": 1,
            "maximum": 3,
            "distribution": "uniform",
        },
        "family_weights": FAMILY_WEIGHTS,
        "join_directive_weights": {"force": 70, "forbid": 20, "memoize": 10},
        "scan_force_probability": 0.8,
        "named_index_probability_given_index_scan": 0.5,
        "tid_scan_sampling": False,
        "disabled_index_sampling": True,
        "setting_values": SETTING_VALUES,
        "postgresql_applicability_filtering": False,
    }
