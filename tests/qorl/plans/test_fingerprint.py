from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from qorl.plans.catalog import TaskCatalog
from qorl.plans.fingerprint import (
    PLAN_FINGERPRINT_VERSION,
    canonical_plan,
    plan_sha256,
    structural_plan_sha256,
    timing_reuse_key,
)
from qorl.plans.schemas import PlanAction
from qorl.sft.schemas import load_json_object, require_list, require_object


class ExplainDocument(BaseModel):
    Plan: dict[str, JsonValue]


class IoTimingProbe(BaseModel):
    name: str
    sql: str
    options: str
    analyze: bool
    document: ExplainDocument


@pytest.mark.parametrize(
    "record_path",
    [
        "parallel/ceb-6a-6a19.json",
        "parallel/ceb-6a-6a267.json",
        "parallel/ceb-6a-6a462.json",
        "parallel/ceb-6a-6a428.json",
        "leading/ceb-9b-7b4fd8123bd3e02940d8f1c8316473ca7f0ebe12.json",
    ],
)
def test_recorded_teacher_estimate_only_changes_are_not_novel(
    repository_root: Path, record_path: str
) -> None:
    path = (
        repository_root
        / "experiments/005-protocol-sft-v2/teacher/records"
        / record_path
    )
    # Read the original plan evidence, not the retired teacher record schema.
    sample = require_object(
        load_json_object(path)["accepted_sample"], "accepted_sample"
    )
    baseline = require_object(sample["default"], "default")
    first = require_object(
        require_list(sample["candidates"], "candidates")[0], "candidate"
    )
    default = ExplainDocument.model_validate(baseline["plain_explain"]).Plan
    candidate = ExplainDocument.model_validate(first["plain_explain"]).Plan
    assert plan_sha256(default) != plan_sha256(candidate)
    assert structural_plan_sha256(default) == structural_plan_sha256(candidate)


@pytest.mark.parametrize(
    "field", ["Startup Cost", "Total Cost", "Plan Rows", "Plan Width"]
)
def test_estimates_change_full_identity_but_not_structure(field: str) -> None:
    first: dict[str, JsonValue] = {
        "Node Type": "Hash Join",
        "Plans": [{"Node Type": "Seq Scan", "Alias": "t", field: 1}],
    }
    second: dict[str, JsonValue] = {
        "Node Type": "Hash Join",
        "Plans": [{"Node Type": "Seq Scan", "Alias": "t", field: 2}],
    }
    assert plan_sha256(first) != plan_sha256(second)
    assert structural_plan_sha256(first) == structural_plan_sha256(second)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Node Type", "Index Only Scan"),
        ("Index Name", "another_index"),
        ("Index Cond", "id > 7"),
        ("Filter", "kind_id = 3"),
        ("Workers Planned", 4),
        ("Parallel Aware", True),
        ("Join Type", "Left"),
        ("Hash Cond", "a.id = b.other_id"),
    ],
)
def test_structural_identity_retains_execution_properties(
    field: str, value: JsonValue
) -> None:
    original: dict[str, JsonValue] = {
        "Node Type": "Index Scan",
        "Index Name": "title_pkey",
        "Index Cond": "id > 5",
        "Filter": "kind_id = 2",
        "Workers Planned": 2,
        "Parallel Aware": False,
        "Join Type": "Inner",
        "Hash Cond": "a.id = b.id",
    }
    changed = {**original, field: value}
    assert structural_plan_sha256(original) != structural_plan_sha256(changed)


def test_child_orientation_is_not_sorted_away() -> None:
    children: list[JsonValue] = [
        {"Node Type": "Seq Scan", "Alias": "a"},
        {"Node Type": "Seq Scan", "Alias": "b"},
    ]
    first: dict[str, JsonValue] = {"Node Type": "Nested Loop", "Plans": children}
    swapped: dict[str, JsonValue] = {
        "Node Type": "Nested Loop",
        "Plans": list(reversed(children)),
    }
    assert structural_plan_sha256(first) != structural_plan_sha256(swapped)
    assert structural_plan_sha256(first) == structural_plan_sha256(
        dict(reversed(list(first.items())))
    )


def test_io_timing_reproduction_has_one_plan_identity() -> None:
    # Unmodified EXPLAIN documents from the PostgreSQL 18.6 sort probe,
    # captured 2026-09-06. SQL and per-probe settings are retained in the fixture.
    probes = TypeAdapter(list[IoTimingProbe]).validate_json(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("io_timing_plans.json")
        .read_text()
    )
    assert {probe.name for probe in probes} == {
        "sort_plain_timing_on",
        "sort_4MB_timing_off",
        "sort_4MB_timing_on",
        "sort_4MB_timing_on_repeat",
    }
    assert len({plan_sha256(probe.document.Plan) for probe in probes}) == 1
    assert len({structural_plan_sha256(probe.document.Plan) for probe in probes}) == 1


@pytest.mark.parametrize("scope", ["Shared", "Local", "Temp"])
@pytest.mark.parametrize("direction", ["Read", "Write"])
@pytest.mark.parametrize("milliseconds", [0.0, 3.5])
def test_scoped_io_times_are_ignored_recursively(
    scope: str, direction: str, milliseconds: float
) -> None:
    field = f"{scope} I/O {direction} Time"
    plain: dict[str, JsonValue] = {
        "Node Type": "Sort",
        "Plans": [{"Node Type": "Seq Scan"}],
    }
    timed: dict[str, JsonValue] = {
        "Node Type": "Sort",
        field: milliseconds,
        "Plans": [{"Node Type": "Seq Scan", field: milliseconds}],
    }
    assert plan_sha256(plain) == plan_sha256(timed)
    assert structural_plan_sha256(plain) == structural_plan_sha256(timed)


def test_execution_reuse_requires_identical_overrides() -> None:
    catalog = TaskCatalog(
        relations=frozenset({"a"}), adjacency={"a": frozenset()}, indexes={}
    )
    empty = PlanAction.from_raw({"version": 1}, catalog)
    setting = PlanAction.from_raw(
        {"version": 1, "settings": {"enable_material": False}}, catalog
    )
    parallel = PlanAction.from_raw(
        {"version": 1, "parallel": [{"relation": "a", "workers": 2, "mode": "soft"}]},
        catalog,
    )
    full = plan_sha256({"Node Type": "Seq Scan", "Alias": "a"})
    assert timing_reuse_key(full) == timing_reuse_key(full, empty)
    assert (
        len(
            {
                timing_reuse_key(full),
                timing_reuse_key(full, setting),
                timing_reuse_key(full, parallel),
            }
        )
        == 3
    )


def explain(*, rows: int = 10, hits: int = 100, reads: int = 5) -> dict:
    return {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "title",
            "Plan Rows": 100,
            "Actual Rows": rows,
            "Actual Loops": 1,
            "Shared Hit Blocks": hits,
            "Shared Read Blocks": reads,
        },
        "Planning Time": 0.2,
        "Execution Time": 1.5,
    }


class TestFingerprint:
    def test_plan_fingerprint_ignores_runtime_observations(self) -> None:
        first = explain(rows=10, hits=100, reads=5)["Plan"]
        first.update(
            {
                "Cache Hits": 10,
                "Disk Usage": 0,
                "Hash Batches": 1,
                "HashAgg Batches": 1,
                "Index Searches": 3,
                "Peak Memory Usage": 12,
                "Subplans Removed": 0,
                "Rows Removed by Join Filter": 4,
                "Sort Space Used": 8,
                "Workers Launched": 2,
            }
        )
        second = explain(rows=20, hits=200, reads=9)["Plan"]
        second.update(
            {
                "Cache Hits": 100,
                "Disk Usage": 8192,
                "Hash Batches": 2,
                "HashAgg Batches": 4,
                "Index Searches": 30,
                "Peak Memory Usage": 24,
                "Subplans Removed": 2,
                "Rows Removed by Join Filter": 40,
                "Sort Space Used": 16,
                "Workers Launched": 1,
            }
        )
        assert plan_sha256(first) == plan_sha256(second)

    def test_plain_and_analyzed_hash_aggregate_have_the_same_fingerprint(self) -> None:
        plain = {
            "Node Type": "Aggregate",
            "Strategy": "Hashed",
            "Partial Mode": "Simple",
            "Parallel Aware": False,
            "Async Capable": False,
            "Startup Cost": 100.0,
            "Total Cost": 120.0,
            "Plan Rows": 10,
            "Plan Width": 16,
            "Group Key": ["title.kind_id"],
            "Planned Partitions": 0,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "title",
                    "Alias": "title",
                    "Startup Cost": 0.0,
                    "Total Cost": 90.0,
                    "Plan Rows": 1000,
                    "Plan Width": 8,
                }
            ],
        }
        analyzed = json.loads(json.dumps(plain))
        analyzed.update(
            {
                "Actual Startup Time": 0.1,
                "Actual Total Time": 1.2,
                "Actual Rows": 10,
                "Actual Loops": 1,
                "HashAgg Batches": 3,
                "Peak Memory Usage": 129,
                "Disk Usage": 456,
                "Shared Hit Blocks": 100,
                "Shared Read Blocks": 5,
            }
        )
        analyzed["Plans"][0].update(
            {
                "Actual Startup Time": 0.01,
                "Actual Total Time": 0.8,
                "Actual Rows": 1000,
                "Actual Loops": 1,
                "Rows Removed by Filter": 4,
                "Shared Hit Blocks": 100,
            }
        )

        assert PLAN_FINGERPRINT_VERSION == 4
        assert canonical_plan(analyzed) == plain
        assert plan_sha256(analyzed) == plan_sha256(plain)

    def test_incremental_sort_runtime_groups_do_not_change_fingerprint(self) -> None:
        plain = {
            "Node Type": "Incremental Sort",
            "Sort Key": ["title.kind_id", "title.id"],
            "Presorted Key": ["title.kind_id"],
            "Plan Rows": 1000,
        }
        analyzed = {
            **plain,
            "Full-sort Groups": {
                "Group Count": 4,
                "Sort Methods Used": ["quicksort"],
                "Sort Space Memory": {
                    "Average Sort Space Used": 27,
                    "Peak Sort Space Used": 27,
                },
            },
            "Pre-sorted Groups": {
                "Group Count": 2,
                "Sort Methods Used": ["external merge"],
                "Sort Space Disk": {
                    "Average Sort Space Used": 128,
                    "Peak Sort Space Used": 256,
                },
            },
        }
        assert plan_sha256(analyzed) == plan_sha256(plain)

    def test_fingerprint_keeps_planner_fields_near_runtime_fields(self) -> None:
        plan = {
            "Node Type": "Gather",
            "Parallel Aware": False,
            "Workers Planned": 2,
            "Plan Rows": 100,
            "Plans": [
                {
                    "Node Type": "Aggregate",
                    "Strategy": "Hashed",
                    "Planned Partitions": 4,
                    "Plan Rows": 10,
                }
            ],
        }
        analyzed = {
            **plan,
            "Workers Launched": 1,
            "Workers": [{"Worker Number": 0, "Actual Rows": 50}],
        }

        assert canonical_plan(analyzed) == plan
        assert plan_sha256(plan) != plan_sha256({**plan, "Workers Planned": 1})
        changed_partitions = json.loads(json.dumps(plan))
        changed_partitions["Plans"][0]["Planned Partitions"] = 2
        assert plan_sha256(plan) != plan_sha256(changed_partitions)

    def test_plan_fingerprint_detects_physical_plan_change(self) -> None:
        first = explain()["Plan"]
        second = {**first, "Node Type": "Index Scan"}
        assert plan_sha256(first) != plan_sha256(second)
