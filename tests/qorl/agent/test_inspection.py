"""The six-tool inspection contract, including evidence and argument boundaries."""

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pytest
from tests.qorl.agent.test_agent import TASK, Database, SqlFixture

from qorl.agent.interface import AgentInterface
from qorl.agent.presentation import PLAN_BYTES, plan_view
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor
from qorl.measure.validation import PlanValidationEvaluator
from qorl.model.schemas import JsonObject, JsonValue
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.inspection import (
    DISTRIBUTION_ITEMS,
    RELATION_BYTES,
    column_statistics,
    inspect_relation,
)
from qorl.postgres.schemas import PostgresIndexes, WorkerAllocation


@dataclass
class MetadataDatabase(Database):
    response: str = '{"exists":false}'
    queries: list[str] = field(default_factory=list[str])

    def admin_sql(self, sql: str) -> str:
        self.queries.append(sql)
        return self.response


@pytest.fixture
def database(repository_root: Path) -> MetadataDatabase:
    config = PostgresConfig.load(
        repository_root / "docker/postgres/configs/000-pgconf-default"
    )
    return MetadataDatabase(config.agent_settings, PostgresIndexes(by_table={}))


def environment(database: MetadataDatabase) -> AgentEnvironment:
    task = TASK.model_copy(
        update={
            "relations": [
                TASK.relations[0],
                TASK.relations[1].model_copy(update={"table": "table_a"}),
            ],
        }
    )
    evaluator = PlanValidationEvaluator[InspectionExecutor](
        database, SqlFixture(), task, default_timeout_ms=5000, max_candidates=1
    )
    evaluator.start()
    return AgentEnvironment(evaluator)


def test_summary_and_detail_are_estimates_with_complete_aliases() -> None:
    plan: JsonObject = {
        "Node Type": "Hash Join",
        "Plan Rows": 10,
        "Startup Cost": 5,
        "Total Cost": 20,
        "Plan Width": 64,
        "Hash Cond": "(a.id = b.id)",
        "Actual Rows": 777,
        "Workers Launched": 9,
        "Plans": [
            {
                "Node Type": "Index Scan",
                "Alias": "a",
                "Relation Name": "same_table",
                "Index Name": "idx_a",
                "Index Cond": "(id > 2)",
                "Filter": "active",
                "Plan Rows": 3,
                "Parent Relationship": "Outer",
                "Actual Rows": 999,
            },
            {
                "Node Type": "Gather Merge",
                "Workers Planned": 2,
                "Parallel Aware": True,
                "Parent Relationship": "Inner",
                "Sort Key": ["b.id"],
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Alias": "b",
                        "Relation Name": "same_table",
                    }
                ],
            },
        ],
    }
    original = json.dumps(plan)
    summary, detail = plan_view(plan, summary=True), plan_view(plan)
    assert summary.nodes[0].leaf_aliases == ["a", "b"]
    assert "Total Cost" not in summary.nodes[0].estimates
    assert detail.nodes[0].estimates["Total Cost"] == 20
    assert detail.nodes[0].estimates["Plan Width"] == 64
    assert detail.nodes[0].child_ids == ["0.0", "0.1"]
    assert detail.nodes[1].estimates["Parent Relationship"] == "Outer"
    assert detail.nodes[1].estimates["Index Cond"] == "(id > 2)"
    assert detail.nodes[2].estimates["Sort Key"] == ["b.id"]
    assert detail.nodes[2].estimates["Workers Planned"] == 2
    assert "Actual Rows" not in detail.model_dump_json()
    assert "Workers Launched" not in detail.model_dump_json()
    assert "planner_units_not_milliseconds" in detail.model_dump_json()
    assert json.dumps(plan) == original


def test_many_aliases_and_subtree_navigation_are_bounded() -> None:
    children: list[JsonValue] = [
        {"Node Type": "Seq Scan", "Alias": f"a{i}", "Plan Rows": i} for i in range(80)
    ]
    plan: JsonObject = {"Node Type": "Append", "Plans": children, "Filter": "x" * 10000}
    summary, detail = plan_view(plan, summary=True), plan_view(plan)
    assert len(summary.nodes) == 16 and summary.omitted_nodes == 65
    assert len(detail.nodes) == 32 and detail.omitted_nodes == 49
    assert len(detail.nodes[0].leaf_aliases) == 80
    assert detail.nodes[0].omitted_fields == ["Filter"]
    assert len(detail.model_dump_json().encode()) <= PLAN_BYTES
    assert json.loads(detail.model_dump_json())["nodes"]
    last = plan_view(plan, node_id=detail.nodes[0].child_ids[-1])
    assert last.nodes[0].leaf_aliases == ["a79"]
    with pytest.raises(ValueError, match="not present"):
        plan_view(plan, node_id="0.999")
    with pytest.raises(PostgresError, match="stored plan"):
        plan_view({"Plans": "corrupt"})


def test_many_alias_join_nodes_keep_all_descendant_aliases() -> None:
    level: list[JsonObject] = [
        {"Node Type": "Seq Scan", "Alias": f"a{i}"} for i in range(40)
    ]
    while len(level) > 1:
        following: list[JsonObject] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                following.append(level[index])
            else:
                following.append(
                    {
                        "Node Type": "Hash Join",
                        "Plans": [level[index], level[index + 1]],
                    }
                )
        level = following
    summary = plan_view(level[0], summary=True)
    assert summary.omitted_nodes == 63
    assert summary.nodes[0].leaf_aliases == sorted(f"a{i}" for i in range(40))
    assert summary.nodes[1].leaf_aliases == sorted(f"a{i}" for i in range(32))
    detail = plan_view(level[0], node_id=summary.nodes[0].child_ids[1])
    assert detail.nodes[0].leaf_aliases == sorted(f"a{i}" for i in range(32, 40))
    assert detail.omitted_nodes == 0


def test_repeated_aliases_share_physical_metadata(database: MetadataDatabase) -> None:
    database.response = json.dumps(
        {
            "exists": True,
            "columns": [{"name": "id", "type": "integer", "nullable": False}],
            "indexes": [
                {"name": "idx", "definition": "CREATE INDEX idx ON table_a(id)"}
            ],
            "extended_statistics": [
                {"name": "ab", "columns": ["id", "value"], "kinds": ["d", "f"]}
            ],
            "estimated_rows": 123,
            "table_bytes": 8192,
            "indexes_bytes": 8192,
            "total_bytes": 16384,
        }
    )
    runtime = environment(database)
    first, _ = runtime.execute("inspect_relation", {"relation": "a"})
    second, _ = runtime.execute("inspect_relation", {"relation": "b"})
    assert first["relation"] == "a" and second["relation"] == "b"
    assert first["table"] == second["table"] == "table_a"
    assert first["query_aliases"] == second["query_aliases"] == ["a", "b"]
    assert first["columns"] == [{"name": "id", "type": "integer", "nullable": False}]
    assert first["extended_statistics"] == [
        {"name": "ab", "columns": ["id", "value"], "kinds": ["d", "f"]}
    ]
    assert len(database.queries) == 1
    assert "dependencies" not in database.queries[0]


def test_missing_and_oversized_metadata() -> None:
    absent = inspect_relation(lambda _: '{"exists":false}', "absent")
    assert not absent.exists and absent.estimated_rows is None and absent.columns == []
    huge = json.dumps(
        {
            "exists": True,
            "indexes": [
                {"name": "big", "definition": "x" * RELATION_BYTES},
            ],
        }
    )
    bounded = inspect_relation(lambda _: huge, "table_a")
    assert bounded.indexes == [] and bounded.omitted_indexes == 1
    assert len(bounded.model_dump_json().encode()) <= RELATION_BYTES
    with pytest.raises(PostgresError, match="invalid inspection metadata"):
        inspect_relation(lambda _: "not JSON", "table_a")


def test_structured_statistics_missing_data_and_omissions() -> None:
    raw = json.dumps(
        {
            "columns": [
                {
                    "column": "id",
                    "status": "available",
                    "n_distinct": -1,
                    "correlation": -0.9,
                    "most_common_values": [1, "x" * 200],
                    "most_common_frequencies": [0.2, 0.1],
                    "histogram_bounds": [0, 10, 20],
                    "common_value_count": 20,
                    "histogram_bound_count": 3,
                },
                {"column": "unanalyzed", "status": "missing_statistics"},
                {"column": "absent", "status": "missing_column"},
            ]
        }
    )
    result = column_statistics(lambda _: raw, "table_a", ["id", "unanalyzed", "absent"])
    stats = result.columns[0]
    assert stats.n_distinct == -1 and stats.correlation == -0.9
    assert stats.most_common_values == [1, None]
    assert stats.most_common_frequencies == [0.2, 0.1]
    assert stats.histogram_bounds == [0, 10, 20]
    assert stats.histogram_bound_positions == [0, 1, 2]
    assert (
        stats.omitted_common_value_indexes == [1] and stats.omitted_common_values == 4
    )
    assert [item.status for item in result.columns] == [
        "available",
        "missing_statistics",
        "missing_column",
    ]
    assert (
        "fraction" in result.n_distinct_meaning
        and "physical" in result.correlation_meaning
    )


@pytest.mark.parametrize("count", [None, 0, 1, 2, 15, 16, 17, 101, 257])
def test_histogram_spans_original_bounds_without_interpolation(
    count: int | None,
) -> None:
    bounds: list[JsonValue] | None = (
        None if count is None else [index**2 for index in range(count)]
    )
    common = list(range(DISTRIBUTION_ITEMS))
    frequencies = [1 / (index + 2) for index in common]
    raw = json.dumps(
        {
            "columns": [
                {
                    "column": "id",
                    "status": "available",
                    "histogram_bounds": bounds,
                    "histogram_bound_count": count or 0,
                    "most_common_values": common,
                    "most_common_frequencies": frequencies,
                    "common_value_count": 100,
                }
            ]
        }
    )
    result = column_statistics(lambda _: raw, "table_a", ["id"])
    stats = result.columns[0]
    positions = stats.histogram_bound_positions
    assert stats.histogram_bound_count == (count or 0)
    assert len(positions) == min(count or 0, DISTRIBUTION_ITEMS)
    if bounds is None:
        assert stats.histogram_bounds is None and positions == []
    elif len(bounds) <= DISTRIBUTION_ITEMS:
        assert stats.histogram_bounds == bounds
        assert positions == list(range(len(bounds)))
    else:
        assert positions[0] == 0 and positions[-1] == len(bounds) - 1
        assert positions == sorted(set(positions))
        gaps = [right - left for left, right in pairwise(positions)]
        assert max(gaps) - min(gaps) <= 1
        assert stats.histogram_bounds == [bounds[index] for index in positions]
    assert stats.omitted_histogram_bounds == (count or 0) - len(positions)
    assert stats.omitted_histogram_bound_indexes == []
    assert stats.most_common_values == common
    assert stats.most_common_frequencies == frequencies
    assert stats.omitted_common_values == 84
    assert "original ordered histogram" in result.histogram_positions_meaning
    assert len(result.model_dump_json().encode()) < RELATION_BYTES


def test_oversized_selected_histogram_value_marks_returned_slot() -> None:
    bounds: list[JsonValue] = list(range(101))
    bounds[6] = "oversized" * 100
    raw = json.dumps(
        {
            "columns": [
                {
                    "column": "id",
                    "status": "available",
                    "histogram_bounds": bounds,
                    "histogram_bound_count": 101,
                }
            ]
        }
    )
    result = column_statistics(lambda _: raw, "table_a", ["id"])
    stats = result.columns[0]
    assert stats.histogram_bound_positions == [
        0,
        6,
        13,
        20,
        26,
        33,
        40,
        46,
        53,
        60,
        66,
        73,
        80,
        86,
        93,
        100,
    ]
    assert stats.histogram_bounds is not None and stats.histogram_bounds[1] is None
    assert stats.histogram_bounds[0] == 0 and stats.histogram_bounds[-1] == 100
    assert stats.omitted_histogram_bound_indexes == [1]
    assert stats.omitted_histogram_bounds == 85
    assert "returned-array slots" in result.histogram_positions_meaning


@pytest.mark.parametrize(
    ("name", "arguments", "field"),
    [
        ("get_column_stats", {"relation": "a", "columns": []}, "columns"),
        ("get_column_stats", {"relation": "a", "columns": ["id"] * 9}, "columns"),
        ("get_column_stats", {"relation": "a", "columns": ["id", "id"]}, "columns"),
        ("get_column_stats", {"relation": "a", "columns": [True]}, "columns.0"),
        ("get_column_stats", {"relation": "a", "columns": ["id;DROP"]}, "columns.0"),
        ("inspect_relation", {"relation": "a", "oops": True}, "oops"),
        ("inspect_relation", {}, "relation"),
        ("inspect_relation", [], "arguments"),
        ("get_plan", {"candidate_id": "default", "node_id": "bad"}, "node_id"),
        ("finish", {"oops": True}, "oops"),
    ],
)
def test_invalid_arguments_are_field_first_without_sql(
    database: MetadataDatabase, name: str, arguments: JsonValue, field: str
) -> None:
    result, finished = environment(database).execute(name, arguments)
    message = result["error"]
    assert isinstance(message, str) and message.startswith(field + ":")
    assert all(
        text not in message
        for text in ("https:", "Arguments", "input_type", "input_value")
    )
    assert not finished and not database.queries and database.explain_calls == 1


@pytest.mark.parametrize(
    ("arguments", "diagnostic", "action"),
    [
        ({}, "action: required", None),
        ({"action": {"version": 1}, "oops": True}, "oops: not allowed", {"version": 1}),
        ([], "arguments: must be an object", None),
        ("raw input", "arguments: must be an object", None),
        (None, "arguments: must be an object", None),
        (7, "arguments: must be an object", None),
    ],
)
def test_rejected_envelopes_retain_original_evidence_and_consume_attempt(
    database: MetadataDatabase, arguments: JsonValue, diagnostic: str, action: JsonValue
) -> None:
    runtime = environment(database)
    result, finished = runtime.execute("evaluate_candidate", arguments)
    candidate = runtime.evaluator.candidates[0]
    assert candidate.candidate_id == "candidate-01"
    assert candidate.rejected_tool_arguments == arguments
    assert candidate.action == action and not candidate.action_valid
    assert result["errors_or_diagnostics"] == [diagnostic]
    assert result["attempts_remaining"] == 0
    assert not finished and database.explain_calls == 1 and database.queries == []


def test_six_tools_resources_and_unavailable_metadata(
    database: MetadataDatabase,
) -> None:
    runtime = environment(database)
    interface = AgentInterface.from_evaluator(runtime.evaluator, 64)
    assert interface.observation.resource_limits.worker is None
    database.allocation = WorkerAllocation(
        cpuset="0-3", physical_core_count=4, memory_bytes=8192
    )
    interface = AgentInterface.from_evaluator(runtime.evaluator, 64)
    assert interface.observation.resource_limits.worker == database.allocation
    assert interface.inspection_turn_limit == 6
    assert [tool.function.name for tool in agent_tools(["a", "b"])] == [
        "inspect_relation",
        "get_column_stats",
        "get_plan",
        "evaluate_candidate",
        "keep_default",
        "finish",
    ]
    result, _ = runtime.execute("inspect_relation", {"relation": "a"})
    assert result["exists"] is False
    database.response = "corrupt"
    with pytest.raises(PostgresError, match="invalid inspection metadata"):
        runtime.execute("get_column_stats", {"relation": "a", "columns": ["id"]})
    invalid_requests: list[tuple[str, JsonObject]] = [
        ("inspect_relation", {"relation": "table_a"}),
        ("get_plan", {"candidate_id": "candidate-99"}),
    ]
    for name, arguments in invalid_requests:
        result, finished = runtime.execute(name, arguments)
        assert "error" in result and not finished
