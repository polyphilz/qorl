from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


class ActionError(ValueError):
    pass


IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
JOIN_METHODS = {"hash", "merge", "nestloop"}
SCAN_METHODS = {"seq", "tid", "index", "index_only", "bitmap"}

BOOLEAN_SETTINGS = {
    "enable_async_append",
    "enable_bitmapscan",
    "enable_distinct_reordering",
    "enable_gathermerge",
    "enable_group_by_reordering",
    "enable_hashagg",
    "enable_hashjoin",
    "enable_incremental_sort",
    "enable_indexonlyscan",
    "enable_indexscan",
    "enable_material",
    "enable_memoize",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_parallel_append",
    "enable_parallel_hash",
    "enable_partition_pruning",
    "enable_partitionwise_aggregate",
    "enable_partitionwise_join",
    "enable_presorted_aggregate",
    "enable_self_join_elimination",
    "enable_seqscan",
    "enable_sort",
    "enable_tidscan",
    "geqo",
}

NUMERIC_SETTINGS = {
    "seq_page_cost": (0.0, 1_000_000.0),
    "random_page_cost": (0.0, 1_000_000.0),
    "cpu_tuple_cost": (0.0, 1_000_000.0),
    "cpu_index_tuple_cost": (0.0, 1_000_000.0),
    "cpu_operator_cost": (0.0, 1_000_000.0),
    "parallel_setup_cost": (0.0, 1_000_000.0),
    "parallel_tuple_cost": (0.0, 1_000_000.0),
    "geqo_selection_bias": (1.5, 2.0),
    "geqo_seed": (0.0, 1.0),
}

INTEGER_SETTINGS = {
    "effective_cache_size": (1, 4_194_304),
    "effective_io_concurrency": (0, 1_000),
    "from_collapse_limit": (1, 32),
    "join_collapse_limit": (1, 32),
    "geqo_threshold": (2, 32),
    "geqo_effort": (1, 10),
    "geqo_pool_size": (0, 10_000),
    "geqo_generations": (0, 10_000),
    "max_parallel_workers_per_gather": (0, 8),
}


@dataclass(frozen=True)
class TaskCatalog:
    relations: frozenset[str]
    adjacency: dict[str, frozenset[str]]
    indexes: dict[str, frozenset[str]]

    @classmethod
    def from_task(
        cls,
        task: dict[str, Any],
        indexes: dict[str, set[str]] | None = None,
    ) -> TaskCatalog:
        relations = frozenset(item["alias"] for item in task["relations"])
        adjacency: dict[str, set[str]] = {relation: set() for relation in relations}
        for edge in task["join_edges"]:
            left, right = edge.split("=", 1)
            left_alias = left.split(":", 1)[0]
            right_alias = right.split(":", 1)[0]
            adjacency[left_alias].add(right_alias)
            adjacency[right_alias].add(left_alias)
        supplied = indexes or {}
        return cls(
            relations=relations,
            adjacency={key: frozenset(value) for key, value in adjacency.items()},
            indexes={key: frozenset(value) for key, value in supplied.items()},
        )

    def require_relations(self, values: Any, label: str) -> list[str]:
        if not isinstance(values, list) or len(values) < 2:
            raise ActionError(f"{label} must contain at least two relations")
        if any(not isinstance(value, str) for value in values):
            raise ActionError(f"{label} must contain relation names")
        if len(values) != len(set(values)):
            raise ActionError(f"{label} contains duplicate relations")
        unknown = set(values) - self.relations
        if unknown:
            raise ActionError(f"{label} contains unknown relations: {sorted(unknown)}")
        selected = set(values)
        visited: set[str] = set()
        frontier = [values[0]]
        while frontier:
            relation = frontier.pop()
            if relation in visited:
                continue
            visited.add(relation)
            frontier.extend((self.adjacency[relation] & selected) - visited)
        if visited != selected:
            raise ActionError(f"{label} is not connected in the query join graph")
        return sorted(values)


def require_object(value: Any, label: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ActionError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ActionError(f"{label} has unknown fields: {sorted(unknown)}")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ActionError(f"{label} must be a list")
    return value


def require_enum(value: Any, allowed: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ActionError(f"{label} must be one of {sorted(allowed)}")
    return value


def require_relation(value: Any, catalog: TaskCatalog, label: str) -> str:
    if not isinstance(value, str) or value not in catalog.relations:
        raise ActionError(f"{label} is not a relation in this query")
    return value


def require_indexes(
    value: Any, relation: str, catalog: TaskCatalog, label: str
) -> list[str]:
    indexes = require_list(value, label)
    if any(not isinstance(index, str) or not IDENTIFIER.fullmatch(index) for index in indexes):
        raise ActionError(f"{label} contains an invalid index name")
    if len(indexes) != len(set(indexes)):
        raise ActionError(f"{label} contains duplicate indexes")
    unknown = set(indexes) - catalog.indexes.get(relation, frozenset())
    if unknown:
        raise ActionError(f"{label} contains unknown indexes: {sorted(unknown)}")
    return sorted(indexes)


def normalize_leading(value: Any, catalog: TaskCatalog) -> dict[str, Any]:
    leaves: list[str] = []

    def visit(
        node: Any, label: str
    ) -> tuple[str | dict[str, Any], frozenset[str]]:
        if isinstance(node, str):
            relation = require_relation(node, catalog, label)
            leaves.append(relation)
            return relation, frozenset({relation})
        item = require_object(node, label, {"left", "right"})
        if set(item) != {"left", "right"}:
            raise ActionError(f"{label} requires left and right")
        left, left_relations = visit(item["left"], f"{label}.left")
        right, right_relations = visit(item["right"], f"{label}.right")
        if not any(
            catalog.adjacency[relation] & right_relations
            for relation in left_relations
        ):
            raise ActionError(f"{label} joins disconnected subtrees")
        return (
            {"left": left, "right": right},
            left_relations | right_relations,
        )

    tree, _ = visit(value, "leading")
    if isinstance(tree, str):
        raise ActionError("leading must contain at least two relations")
    if len(leaves) != len(set(leaves)):
        raise ActionError("leading contains duplicate relations")
    if set(leaves) != catalog.relations:
        raise ActionError("leading must contain every query relation exactly once")
    return tree


def normalize_settings(value: Any) -> dict[str, bool | int | float]:
    settings = require_object(value or {}, "settings", BOOLEAN_SETTINGS | NUMERIC_SETTINGS.keys() | INTEGER_SETTINGS.keys())
    normalized: dict[str, bool | int | float] = {}
    for name, setting in settings.items():
        if name in BOOLEAN_SETTINGS:
            if not isinstance(setting, bool):
                raise ActionError(f"settings.{name} must be boolean")
        elif name in INTEGER_SETTINGS:
            if isinstance(setting, bool) or not isinstance(setting, int):
                raise ActionError(f"settings.{name} must be an integer")
            lower, upper = INTEGER_SETTINGS[name]
            if not lower <= setting <= upper:
                raise ActionError(f"settings.{name} is outside [{lower}, {upper}]")
        else:
            if isinstance(setting, bool) or not isinstance(setting, (int, float)):
                raise ActionError(f"settings.{name} must be numeric")
            lower, upper = NUMERIC_SETTINGS[name]
            if not math.isfinite(setting) or not lower <= setting <= upper:
                raise ActionError(f"settings.{name} is outside [{lower}, {upper}]")
        normalized[name] = setting
    return dict(sorted(normalized.items()))


def normalize_action(value: Any, catalog: TaskCatalog) -> dict[str, Any]:
    action = require_object(
        value,
        "action",
        {
            "version",
            "leading",
            "joins",
            "scans",
            "disabled_indexes",
            "row_corrections",
            "parallel",
            "settings",
        },
    )
    if action.get("version") != 1:
        raise ActionError("action.version must equal 1")

    normalized: dict[str, Any] = {"version": 1}
    if action.get("leading") is not None:
        normalized["leading"] = normalize_leading(action["leading"], catalog)

    joins: list[dict[str, Any]] = []
    join_targets: set[tuple[str, ...]] = set()
    for index, raw in enumerate(require_list(action.get("joins"), "joins")):
        label = f"joins[{index}]"
        item = require_object(raw, label, {"relations", "force", "forbid", "memoize"})
        relations = catalog.require_relations(item.get("relations"), f"{label}.relations")
        target = tuple(relations)
        if target in join_targets:
            raise ActionError(f"{label} duplicates another join target")
        join_targets.add(target)
        force = require_enum(item.get("force", "auto"), JOIN_METHODS | {"auto"}, f"{label}.force")
        forbid = require_list(item.get("forbid"), f"{label}.forbid")
        if len(forbid) != len(set(forbid)) or any(method not in JOIN_METHODS for method in forbid):
            raise ActionError(f"{label}.forbid contains invalid or duplicate methods")
        if set(forbid) == JOIN_METHODS:
            raise ActionError(f"{label}.forbid cannot disable every join method")
        if force in forbid:
            raise ActionError(f"{label} both forces and forbids {force}")
        memoize = require_enum(item.get("memoize", "auto"), {"auto", "force", "forbid"}, f"{label}.memoize")
        if force == "auto" and not forbid and memoize == "auto":
            raise ActionError(f"{label} does not request any steering")
        joins.append({"relations": relations, "force": force, "forbid": sorted(forbid), "memoize": memoize})
    if joins:
        normalized["joins"] = sorted(joins, key=lambda item: item["relations"])

    scans: list[dict[str, Any]] = []
    scan_relations: set[str] = set()
    for index, raw in enumerate(require_list(action.get("scans"), "scans")):
        label = f"scans[{index}]"
        item = require_object(raw, label, {"relation", "force", "forbid", "indexes"})
        relation = require_relation(item.get("relation"), catalog, f"{label}.relation")
        if relation in scan_relations:
            raise ActionError(f"{label} duplicates another scan target")
        scan_relations.add(relation)
        force = require_enum(item.get("force", "auto"), SCAN_METHODS | {"auto"}, f"{label}.force")
        forbid = require_list(item.get("forbid"), f"{label}.forbid")
        if len(forbid) != len(set(forbid)) or any(method not in SCAN_METHODS for method in forbid):
            raise ActionError(f"{label}.forbid contains invalid or duplicate methods")
        if set(forbid) == SCAN_METHODS:
            raise ActionError(f"{label}.forbid cannot disable every scan method")
        if force in forbid:
            raise ActionError(f"{label} both forces and forbids {force}")
        indexes = require_indexes(item.get("indexes"), relation, catalog, f"{label}.indexes")
        if indexes and force not in {"index", "index_only", "bitmap"}:
            raise ActionError(f"{label}.indexes requires an index-based forced scan")
        if force == "auto" and not forbid:
            raise ActionError(f"{label} does not request any steering")
        scans.append({"relation": relation, "force": force, "forbid": sorted(forbid), "indexes": indexes})
    if scans:
        normalized["scans"] = sorted(scans, key=lambda item: item["relation"])

    disabled_indexes: list[dict[str, Any]] = []
    disabled_relations: set[str] = set()
    for index, raw in enumerate(
        require_list(action.get("disabled_indexes"), "disabled_indexes")
    ):
        label = f"disabled_indexes[{index}]"
        item = require_object(raw, label, {"relation", "indexes"})
        relation = require_relation(item.get("relation"), catalog, f"{label}.relation")
        if relation in disabled_relations:
            raise ActionError(f"{label} duplicates another disabled-index target")
        disabled_relations.add(relation)
        indexes = require_indexes(
            item.get("indexes"), relation, catalog, f"{label}.indexes"
        )
        if not indexes:
            raise ActionError(f"{label}.indexes must contain at least one index")
        disabled_indexes.append({"relation": relation, "indexes": indexes})
    if disabled_indexes:
        normalized["disabled_indexes"] = sorted(
            disabled_indexes, key=lambda item: item["relation"]
        )

    corrections: list[dict[str, Any]] = []
    correction_targets: set[tuple[str, ...]] = set()
    for index, raw in enumerate(require_list(action.get("row_corrections"), "row_corrections")):
        label = f"row_corrections[{index}]"
        item = require_object(raw, label, {"relations", "mode", "value"})
        relations = catalog.require_relations(item.get("relations"), f"{label}.relations")
        target = tuple(relations)
        if target in correction_targets:
            raise ActionError(f"{label} duplicates another row-correction target")
        correction_targets.add(target)
        mode = require_enum(item.get("mode"), {"absolute", "add", "subtract", "multiply"}, f"{label}.mode")
        number = item.get("value")
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ActionError(f"{label}.value must be finite and numeric")
        upper = 1_000.0 if mode == "multiply" else 1_000_000_000_000.0
        lower = 0.001 if mode == "multiply" else 1.0
        if not lower <= number <= upper:
            raise ActionError(f"{label}.value is outside [{lower}, {upper}]")
        corrections.append({"relations": relations, "mode": mode, "value": number})
    if corrections:
        normalized["row_corrections"] = sorted(corrections, key=lambda item: item["relations"])

    parallel: list[dict[str, Any]] = []
    parallel_relations: set[str] = set()
    for index, raw in enumerate(require_list(action.get("parallel"), "parallel")):
        label = f"parallel[{index}]"
        item = require_object(raw, label, {"relation", "workers", "mode"})
        relation = require_relation(item.get("relation"), catalog, f"{label}.relation")
        if relation in parallel_relations:
            raise ActionError(f"{label} duplicates another parallel target")
        parallel_relations.add(relation)
        workers = item.get("workers")
        if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 8:
            raise ActionError(f"{label}.workers must be an integer from 0 through 8")
        mode = require_enum(item.get("mode", "soft"), {"soft", "hard"}, f"{label}.mode")
        parallel.append({"relation": relation, "workers": workers, "mode": mode})
    if parallel:
        normalized["parallel"] = sorted(parallel, key=lambda item: item["relation"])

    settings = normalize_settings(action.get("settings"))
    join_setting = {
        "hash": "enable_hashjoin",
        "merge": "enable_mergejoin",
        "nestloop": "enable_nestloop",
    }
    scan_setting = {
        "seq": "enable_seqscan",
        "tid": "enable_tidscan",
        "index": "enable_indexscan",
        "index_only": "enable_indexonlyscan",
        "bitmap": "enable_bitmapscan",
    }
    for item in joins:
        setting = join_setting.get(item["force"])
        if setting and settings.get(setting) is False:
            raise ActionError(f"action both forces {item['force']} and disables {setting}")
    for item in scans:
        setting = scan_setting.get(item["force"])
        if setting and settings.get(setting) is False:
            raise ActionError(f"action both forces {item['force']} and disables {setting}")
        disabled = next(
            (
                set(target["indexes"])
                for target in disabled_indexes
                if target["relation"] == item["relation"]
            ),
            set(),
        )
        if disabled & set(item["indexes"]):
            raise ActionError(
                f"action both forces and disables an index on {item['relation']}"
            )
        if (
            item["force"] in {"index", "index_only", "bitmap"}
            and disabled
            and disabled == set(catalog.indexes.get(item["relation"], ()))
        ):
            raise ActionError(
                f"action disables every index for forced scan on {item['relation']}"
            )
    if any(item["workers"] > 0 for item in parallel) and settings.get(
        "max_parallel_workers_per_gather"
    ) == 0:
        raise ActionError("action both requests parallelism and disables parallel workers")
    if settings:
        normalized["settings"] = settings
    return normalized


def format_number(value: int | float) -> str:
    return format(value, ".15g")


def render_leading(tree: dict[str, Any]) -> str:
    left = tree["left"]
    right = tree["right"]
    if isinstance(left, str) and isinstance(right, str):
        return f"{left} {right}"

    def render(node: str | dict[str, Any]) -> str:
        if isinstance(node, str):
            return node
        return f"({render(node['left'])} {render(node['right'])})"

    return render(tree)


def compile_action(value: Any, catalog: TaskCatalog) -> tuple[dict[str, Any], str]:
    action = normalize_action(value, catalog)
    hints: list[str] = []
    if "leading" in action:
        hints.append(f"Leading({render_leading(action['leading'])})")

    join_names = {"hash": "HashJoin", "merge": "MergeJoin", "nestloop": "NestLoop"}
    for join in action.get("joins", []):
        relations = " ".join(join["relations"])
        if join["force"] != "auto":
            hints.append(f"{join_names[join['force']]}({relations})")
        for method in join["forbid"]:
            hints.append(f"No{join_names[method]}({relations})")
        if join["memoize"] != "auto":
            prefix = "" if join["memoize"] == "force" else "No"
            hints.append(f"{prefix}Memoize({relations})")

    scan_names = {
        "seq": "SeqScan",
        "tid": "TidScan",
        "index": "IndexScan",
        "index_only": "IndexOnlyScan",
        "bitmap": "BitmapScan",
    }
    for scan in action.get("scans", []):
        arguments = " ".join([scan["relation"], *scan["indexes"]])
        if scan["force"] != "auto":
            hints.append(f"{scan_names[scan['force']]}({arguments})")
        for method in scan["forbid"]:
            hints.append(f"No{scan_names[method]}({scan['relation']})")

    for item in action.get("disabled_indexes", []):
        arguments = " ".join([item["relation"], *item["indexes"]])
        hints.append(f"DisableIndex({arguments})")

    correction_prefix = {"absolute": "#", "add": "+", "subtract": "-", "multiply": "*"}
    for item in action.get("row_corrections", []):
        relations = " ".join(item["relations"])
        correction = correction_prefix[item["mode"]] + format_number(item["value"])
        hints.append(f"Rows({relations} {correction})")

    for item in action.get("parallel", []):
        hints.append(f"Parallel({item['relation']} {item['workers']} {item['mode']})")

    for name, value in action.get("settings", {}).items():
        rendered = ("on" if value else "off") if isinstance(value, bool) else format_number(value)
        hints.append(f"Set({name} {rendered})")

    comment = f"/*+ {' '.join(hints)} */" if hints else ""
    return action, comment
