from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from qorl.plans.schemas import (
    AUTO,
    JoinMethod,
    MemoizeMode,
    ParallelMode,
    ScanMethod,
)


@dataclass(frozen=True)
class HintDiagnostics:
    used: str
    not_used: str
    duplicate: str
    error: str


@dataclass(frozen=True)
class Verification:
    valid: bool
    errors: tuple[str, ...]


HINT_DUMP = re.compile(
    r"HintStateDump: \{used hints:(.*?)\}, "
    r"\{not used hints:(.*?)\}, "
    r"\{duplicate hints:(.*?)\}, "
    r"\{error hints:(.*?)\}"
)

JOIN_METHODS = {
    "Hash Join": JoinMethod.HASH.value,
    "Merge Join": JoinMethod.MERGE.value,
    "Nested Loop": JoinMethod.NESTLOOP.value,
}

SCAN_METHODS = {
    "Seq Scan": ScanMethod.SEQ.value,
    "Index Scan": ScanMethod.INDEX.value,
    "Index Only Scan": ScanMethod.INDEX_ONLY.value,
    "Bitmap Heap Scan": ScanMethod.BITMAP.value,
}
MIN_JOIN_CHILDREN = 2


def parse_hint_diagnostics(stderr: str) -> HintDiagnostics | None:
    matches = list(HINT_DUMP.finditer(stderr))
    if not matches:
        return None
    used, not_used, duplicate, error = matches[-1].groups()
    return HintDiagnostics(used, not_used, duplicate, error)


def hint_status(stderr: str) -> dict[str, str] | None:
    diagnostics = parse_hint_diagnostics(stderr)
    if diagnostics is None:
        return None
    return {
        "used": diagnostics.used,
        "not_used": diagnostics.not_used,
        "duplicate": diagnostics.duplicate,
        "error": diagnostics.error,
    }


def nodes(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield plan
    for child in plan.get("Plans", []):
        yield from nodes(child)


def relation_set(plan: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        node["Alias"] for node in nodes(plan) if isinstance(node.get("Alias"), str)
    )


def matching_join(plan: dict[str, Any], relations: list[str]) -> dict[str, Any] | None:
    target = frozenset(relations)
    return next(
        (
            node
            for node in nodes(plan)
            if node.get("Node Type") in JOIN_METHODS and relation_set(node) == target
        ),
        None,
    )


def matching_scan(plan: dict[str, Any], relation: str) -> dict[str, Any] | None:
    return next(
        (
            node
            for node in nodes(plan)
            if node.get("Alias") == relation and node.get("Node Type") in SCAN_METHODS
        ),
        None,
    )


def plan_join_tree(plan: dict[str, Any]) -> str | tuple[Any, Any] | None:
    if plan.get("Node Type") in JOIN_METHODS:
        children = plan.get("Plans", [])
        if len(children) >= MIN_JOIN_CHILDREN:
            return plan_join_tree(children[0]), plan_join_tree(children[1])
    if plan.get("Node Type") in SCAN_METHODS and isinstance(plan.get("Alias"), str):
        return plan["Alias"]
    child_trees = [
        tree
        for child in plan.get("Plans", [])
        if (tree := plan_join_tree(child)) is not None
    ]
    return child_trees[0] if len(child_trees) == 1 else None


def action_join_tree(tree: str | dict[str, Any]) -> str | tuple[Any, Any]:
    if isinstance(tree, str):
        return tree
    return action_join_tree(tree["left"]), action_join_tree(tree["right"])


def contains_node(plan: dict[str, Any], node_type: str) -> bool:
    return any(node.get("Node Type") == node_type for node in nodes(plan))


def memoized_inner(join: dict[str, Any]) -> bool:
    children = join.get("Plans", [])
    return (
        len(children) >= MIN_JOIN_CHILDREN and children[1].get("Node Type") == "Memoize"
    )


def index_names(plan: dict[str, Any]) -> set[str]:
    return {
        node["Index Name"]
        for node in nodes(plan)
        if isinstance(node.get("Index Name"), str)
    }


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "Node Type",
        "Join Type",
        "Relation Name",
        "Alias",
        "Index Name",
        "Plan Rows",
        "Actual Rows",
        "Actual Loops",
        "Shared Hit Blocks",
        "Shared Read Blocks",
        "Temp Read Blocks",
        "Temp Written Blocks",
    )
    result = {key: plan[key] for key in keep if key in plan}
    if plan.get("Plans"):
        result["Plans"] = [compact_plan(child) for child in plan["Plans"]]
    return result


def verify_action(
    action: dict[str, Any], plan: dict[str, Any], stderr: str
) -> Verification:
    errors: list[str] = []
    diagnostics = parse_hint_diagnostics(stderr)
    if diagnostics is None and len(action) > 1:
        errors.append("pg_hint_plan did not emit a HintStateDump")
    elif diagnostics is not None:
        for label, value in (
            ("not used", diagnostics.not_used),
            ("duplicate", diagnostics.duplicate),
            ("error", diagnostics.error),
        ):
            if value != "(none)":
                errors.append(f"pg_hint_plan reported {label} hints: {value}")

    if "leading" in action:
        expected = action_join_tree(action["leading"])
        actual = plan_join_tree(plan)
        if expected != actual:
            errors.append(
                f"leading join tree differs: expected={expected} actual={actual}"
            )

    for item in action.get("joins", []):
        join = matching_join(plan, item["relations"])
        label = " ".join(item["relations"])
        if join is None:
            errors.append(f"join target does not exist in the plan: {label}")
            continue
        method = JOIN_METHODS[join["Node Type"]]
        if item["force"] != AUTO and method != item["force"]:
            errors.append(f"join {label} uses {method}, not {item['force']}")
        if method in item["forbid"]:
            errors.append(f"join {label} uses forbidden method {method}")
        memoized = memoized_inner(join)
        if item["memoize"] == MemoizeMode.FORCE and not memoized:
            errors.append(f"join {label} is not memoized")
        if item["memoize"] == MemoizeMode.FORBID and memoized:
            errors.append(f"join {label} uses forbidden memoization")

    for item in action.get("scans", []):
        scan = matching_scan(plan, item["relation"])
        if scan is None:
            errors.append(f"scan target does not exist in the plan: {item['relation']}")
            continue
        method = SCAN_METHODS[scan["Node Type"]]
        if item["force"] != AUTO and method != item["force"]:
            errors.append(f"scan {item['relation']} uses {method}, not {item['force']}")
        if method in item["forbid"]:
            errors.append(f"scan {item['relation']} uses forbidden method {method}")
        if item["indexes"]:
            used = index_names(scan)
            if not used or not used <= set(item["indexes"]):
                errors.append(
                    f"scan {item['relation']} uses unexpected indexes: {sorted(used)}"
                )

    for item in action.get("disabled_indexes", []):
        scan = matching_scan(plan, item["relation"])
        if scan is None:
            errors.append(
                f"disabled-index target does not exist in the plan: {item['relation']}"
            )
            continue
        used = index_names(scan)
        forbidden = used & set(item["indexes"])
        if forbidden:
            errors.append(
                f"scan {item['relation']} uses disabled indexes: {sorted(forbidden)}"
            )

    for item in action.get("parallel", []):
        scan = matching_scan(plan, item["relation"])
        if scan is None:
            errors.append(
                f"parallel target does not exist in the plan: {item['relation']}"
            )
            continue
        parallel = bool(scan.get("Parallel Aware"))
        if item["workers"] == 0 and parallel:
            errors.append(f"scan {item['relation']} is unexpectedly parallel")
        if item["workers"] > 0 and item["mode"] == ParallelMode.HARD and not parallel:
            errors.append(f"scan {item['relation']} is not parallel")
        if item["workers"] > 0 and item["mode"] == ParallelMode.HARD and parallel:
            worker_counts = {
                node["Workers Planned"]
                for node in nodes(plan)
                if node.get("Node Type") in {"Gather", "Gather Merge"}
                and isinstance(node.get("Workers Planned"), int)
                and item["relation"] in relation_set(node)
            }
            if item["workers"] not in worker_counts:
                errors.append(
                    f"scan {item['relation']} does not plan exactly "
                    f"{item['workers']} workers"
                )

    return Verification(not errors, tuple(errors))
