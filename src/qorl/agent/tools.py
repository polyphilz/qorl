from __future__ import annotations

import json
from typing import Any

from qorl.action import IDENTIFIER, plan_action_schema
from qorl.plan import compact_plan
from qorl.rollout import RolloutEvaluator


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
        "evaluate_candidate",
        "Validate, execute, and measure one self-contained PlanAction.",
        {"action": action},
        ["action"],
    )
    evaluate["function"]["parameters"]["$defs"] = definitions
    return [
        function(
            "describe_table",
            "Return columns and PostgreSQL types for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            "list_indexes",
            "Return index names and definitions for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            "get_column_stats",
            "Return pg_stats values for one column.",
            {"relation": relation, "column": {"type": "string"}},
            ["relation", "column"],
        ),
        function(
            "get_relation_size",
            "Return table, index, and total bytes plus estimated rows.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            "get_extended_stats",
            "Return extended-statistics objects for one query relation.",
            {"relation": relation},
            ["relation"],
        ),
        function(
            "get_plan",
            "Return the compact physical plan for default or a server-issued candidate ID.",
            {"candidate_id": {"type": "string"}},
            ["candidate_id"],
        ),
        evaluate,
        function(
            "keep_default",
            (
                "End the rollout immediately and keep PostgreSQL's default "
                "plan. Available only before submitting a candidate."
            ),
            {},
        ),
        function(
            "finish",
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


class AgentEnvironment:
    def __init__(self, evaluator: RolloutEvaluator) -> None:
        self.evaluator = evaluator
        self.worker = evaluator.worker
        self.tables = {
            item["alias"]: item["table"] for item in evaluator.task["relations"]
        }

    @staticmethod
    def literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def table(self, arguments: dict[str, Any]) -> tuple[str, str]:
        alias = arguments.get("relation")
        if alias not in self.tables:
            raise ValueError("relation is not part of this query")
        return alias, self.tables[alias]

    def query_json(self, sql: str) -> Any:
        output = self.worker.admin_sql(sql).strip()
        return json.loads(output) if output else None

    def describe_table(self, arguments: dict[str, Any]) -> Any:
        alias, table = self.table(arguments)
        name = self.literal(f"public.{table}")
        columns = self.query_json(
            "SELECT COALESCE(json_agg(json_build_object("
            "'name', attname, 'type', format_type(atttypid, atttypmod), "
            "'nullable', NOT attnotnull) ORDER BY attnum), '[]'::json) "
            "FROM pg_attribute WHERE attrelid = to_regclass("
            f"{name}) AND attnum > 0 AND NOT attisdropped;"
        )
        return {"relation": alias, "table": table, "columns": columns}

    def list_indexes(self, arguments: dict[str, Any]) -> Any:
        alias, table = self.table(arguments)
        name = self.literal(table)
        indexes = self.query_json(
            "SELECT COALESCE(json_agg(json_build_object("
            "'name', indexname, 'definition', indexdef) ORDER BY indexname), "
            "'[]'::json) FROM pg_indexes WHERE schemaname = 'public' "
            f"AND tablename = {name};"
        )
        return {"relation": alias, "table": table, "indexes": indexes}

    def get_column_stats(self, arguments: dict[str, Any]) -> Any:
        alias, table = self.table(arguments)
        column = arguments.get("column")
        if not isinstance(column, str) or not IDENTIFIER.fullmatch(column):
            raise ValueError("column is not a valid identifier")
        stats = self.query_json(
            "SELECT json_build_object("
            "'null_fraction', null_frac, 'average_width', avg_width, "
            "'distinct_values', n_distinct, "
            "'most_common_values', most_common_vals::text, "
            "'most_common_frequencies', most_common_freqs, "
            "'histogram_bounds', histogram_bounds::text, "
            "'correlation', correlation) FROM pg_stats "
            f"WHERE schemaname = 'public' AND tablename = {self.literal(table)} "
            f"AND attname = {self.literal(column)};"
        )
        return {
            "relation": alias,
            "table": table,
            "column": column,
            "stats": stats,
        }

    def get_relation_size(self, arguments: dict[str, Any]) -> Any:
        alias, table = self.table(arguments)
        name = self.literal(f"public.{table}")
        size = self.query_json(
            "SELECT json_build_object("
            "'table_bytes', pg_table_size(c.oid), "
            "'indexes_bytes', pg_indexes_size(c.oid), "
            "'total_bytes', pg_total_relation_size(c.oid), "
            "'estimated_rows', c.reltuples) FROM pg_class c "
            f"WHERE c.oid = to_regclass({name});"
        )
        return {"relation": alias, "table": table, **(size or {})}

    def get_extended_stats(self, arguments: dict[str, Any]) -> Any:
        alias, table = self.table(arguments)
        stats = self.query_json(
            "SELECT COALESCE(json_agg(json_build_object("
            "'name', statistics_name, 'columns', attnames, 'kinds', kinds) "
            "ORDER BY statistics_name), '[]'::json) FROM pg_stats_ext "
            f"WHERE schemaname = 'public' AND tablename = {self.literal(table)};"
        )
        return {
            "relation": alias,
            "table": table,
            "extended_statistics": stats,
        }

    def get_plan(self, arguments: dict[str, Any]) -> Any:
        candidate_id = arguments.get("candidate_id")
        if candidate_id == "default":
            plan = self.evaluator.default["plain_explain"]
            return {"Plan": compact_plan(plan["Plan"])}
        candidate = next(
            (
                item
                for item in self.evaluator.candidates
                if item["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("candidate_id was not issued by the server")
        plan = candidate.get("plain_explain") or candidate.get(
            "measured_explain_analyze"
        )
        if plan is None:
            raise ValueError("candidate has no PostgreSQL plan")
        return {"Plan": compact_plan(plan["Plan"])}

    def execute(self, name: str, arguments: Any) -> tuple[Any, bool]:
        if not isinstance(arguments, dict):
            if name == "evaluate_candidate":
                return (
                    candidate_feedback(self.evaluator.evaluate(arguments)),
                    False,
                )
            return {"error": "tool arguments must be an object"}, False
        try:
            if name == "evaluate_candidate":
                return candidate_feedback(
                    self.evaluator.evaluate(arguments.get("action"))
                ), False
            if name == "keep_default":
                return self.evaluator.keep_default(), True
            if name == "finish":
                if not self.evaluator.candidates:
                    raise RuntimeError(
                        "finish requires a candidate; use keep_default to "
                        "keep PostgreSQL's plan"
                    )
                return {"status": "finished"}, True
            methods = {
                "describe_table": self.describe_table,
                "list_indexes": self.list_indexes,
                "get_column_stats": self.get_column_stats,
                "get_relation_size": self.get_relation_size,
                "get_extended_stats": self.get_extended_stats,
                "get_plan": self.get_plan,
            }
            method = methods.get(name)
            if method is None:
                raise ValueError("unknown tool")
            return method(arguments), False
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            return {"error": str(error)}, False
