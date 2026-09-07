from __future__ import annotations

from pydantic import JsonValue

from qorl.plans.schemas import PlanAction
from qorl.util.hashing import sha256_json

PLAN_FINGERPRINT_VERSION = 4

ESTIMATE_PLAN_KEYS = frozenset(
    {"Startup Cost", "Total Cost", "Plan Rows", "Plan Width"}
)

RUNTIME_PLAN_KEYS = {
    "Cache Evictions",
    "Cache Hits",
    "Cache Misses",
    "Cache Overflows",
    "Conflicting Tuples",
    "Disk Usage",
    "Full-sort Groups",
    "Hash Batches",
    "HashAgg Batches",
    "Hash Buckets",
    "Heap Fetches",
    "I/O Read Time",
    "I/O Write Time",
    "Shared I/O Read Time",
    "Shared I/O Write Time",
    "Local I/O Read Time",
    "Local I/O Write Time",
    "Temp I/O Read Time",
    "Temp I/O Write Time",
    "Index Searches",
    "Maximum Storage",
    "Memory Usage",
    "Original Hash Batches",
    "Original Hash Buckets",
    "Peak Memory Usage",
    "Pre-sorted Groups",
    "Sort Method",
    "Sort Space Type",
    "Sort Space Used",
    "Storage",
    "Subplans Removed",
    "Tuples Deleted",
    "Tuples Inserted",
    "Tuples Updated",
    "Workers",
    "Workers Launched",
}
RUNTIME_PLAN_PREFIXES = ("Actual ", "Rows Removed by ", "WAL ")


def canonical_plan(value: JsonValue, *, structural: bool = False) -> JsonValue:
    """Remove observations recursively; structural identity also ignores estimates.

    Child order, predicates, operators and planned execution properties are retained.
    The input is the EXPLAIN document's Plan subtree, not its statement-level metadata.
    """
    if isinstance(value, list):
        return [canonical_plan(item, structural=structural) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: canonical_plan(item, structural=structural)
        for key, item in value.items()
        if key not in RUNTIME_PLAN_KEYS
        and not key.startswith(RUNTIME_PLAN_PREFIXES)
        and not key.endswith(" Blocks")
        and not (structural and key in ESTIMATE_PLAN_KEYS)
    }


def plan_sha256(plan: dict[str, JsonValue]) -> str:
    """Identify the full planned tree, including estimates but excluding observations."""
    return sha256_json(canonical_plan(plan))


def structural_plan_sha256(plan: dict[str, JsonValue]) -> str:
    """Identify physical-plan novelty, independently of cost and cardinality estimates."""
    return sha256_json(canonical_plan(plan, structural=True))


def timing_reuse_key(plan_hash: str, action: PlanAction | None = None) -> str:
    """Conservatively key reuse within one task and fixed PostgreSQL/pool configuration.

    Equal structure is insufficient. Require the full planned tree and identical
    setting/parallel overrides, including planner-only settings. This can miss safe
    reuse. A match establishes reuse eligibility, not identical runtimes. This key
    is not comparable across tasks, fixtures, or database/resource configurations.
    """
    key_input: dict[str, JsonValue] = {"plan_sha256": plan_hash, "parallel": []}
    if action is not None:
        key_input["parallel"] = [
            request.model_dump(mode="json") for request in action.parallel
        ]
        if action.settings.model_fields_set:
            key_input["settings"] = action.settings.model_dump(
                mode="json", exclude_none=True
            )
    return sha256_json(key_input)
