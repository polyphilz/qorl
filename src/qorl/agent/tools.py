from __future__ import annotations

from typing import Any

from qorl.agent.types import ToolName
from qorl.plans.action import plan_action_schema


def function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required or [],
            },
        },
    }


def agent_tools(relations: list[str]) -> list[dict[str, Any]]:
    relation = {"type": "string", "enum": relations}
    action = plan_action_schema(relations)
    definitions = action.pop("$defs")
    evaluate = function(
        ToolName.EVALUATE_CANDIDATE.value,
        "Validate, execute, and measure one self-contained PlanAction.",
        {"action": action},
        ["action"],
    )
    evaluate["function"]["parameters"]["$defs"] = definitions
    return [
        function(
            ToolName.DESCRIBE_TABLE.value,
            "Return columns and PostgreSQL types for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            ToolName.LIST_INDEXES.value,
            "Return index names and definitions for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            ToolName.GET_COLUMN_STATS.value,
            "Return pg_stats values for one column.",
            {"relation": relation, "column": {"type": "string"}},
            ["relation", "column"],
        ),
        function(
            ToolName.GET_RELATION_SIZE.value,
            "Return table, index, and total bytes plus estimated rows.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            ToolName.GET_EXTENDED_STATS.value,
            "Return extended-statistics objects for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            ToolName.GET_PLAN.value,
            "Return the compact physical plan for default or a server-issued candidate ID.",
            {"candidate_id": {"type": "string"}},
            ["candidate_id"],
        ),
        evaluate,
        function(
            ToolName.KEEP_DEFAULT.value,
            (
                "End the rollout immediately and keep PostgreSQL's default "
                "plan. Available only before submitting a candidate."
            ),
            {},
        ),
        function(
            ToolName.FINISH.value,
            "End the search and benchmark the best valid candidate.",
            {},
        ),
    ]


def candidate_feedback(candidate: dict[str, Any]) -> dict[str, Any]:
    measurements = candidate.get("provisional_measurements", [])
    return {
        "candidate_id": candidate["candidate_id"],
        "action_valid": candidate["action_valid"],
        "constraints_satisfied": candidate["constraints_satisfied"],
        "compiled_hint": candidate["compiled_hint"],
        "duplicate_of": candidate["duplicate_of"],
        "plan_sha256": candidate["plan_sha256"],
        "compact_plan": candidate.get("compact_plan"),
        "planning_time_ms": [item["planning_time_ms"] for item in measurements],
        "execution_time_ms": [item["execution_time_ms"] for item in measurements],
        "provisional_speedup": candidate["provisional_speedup"],
        "execution_timed_out": candidate.get("execution_timed_out", False),
        "timeout_ms": candidate.get("timeout_ms"),
        "errors_or_diagnostics": candidate["errors_or_diagnostics"],
        "attempts_remaining": candidate["attempts_remaining"],
    }
