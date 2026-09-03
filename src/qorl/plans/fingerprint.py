from __future__ import annotations

import hashlib
import json
from typing import Any

PLAN_FINGERPRINT_VERSION = 3

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


def canonical_plan(value: Any) -> Any:
    if isinstance(value, list):
        return [canonical_plan(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: canonical_plan(item)
        for key, item in value.items()
        if key not in RUNTIME_PLAN_KEYS
        and not key.startswith(RUNTIME_PLAN_PREFIXES)
        and not key.endswith(" Blocks")
    }


def plan_sha256(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_plan(plan), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
