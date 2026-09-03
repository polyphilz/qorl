from __future__ import annotations

from typing import Any

from qorl.agent.types import ToolName
from qorl.plans.schemas import PlanAction


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
    action = PlanAction.tool_schema(relations)
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
