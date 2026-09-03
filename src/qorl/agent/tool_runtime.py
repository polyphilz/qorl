from __future__ import annotations

import json
from typing import Any

from qorl.agent.types import InspectionExecutor, ToolName
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import ToolResultStatus
from qorl.plans.catalog import IDENTIFIER
from qorl.plans.verify import compact_plan


class AgentEnvironment:
    def __init__(self, evaluator: RolloutEvaluator[InspectionExecutor]) -> None:
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
        if not isinstance(alias, str) or alias not in self.tables:
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
            if self.evaluator.default is None:
                raise RuntimeError("rollout baseline has not been started")
            plan = self.evaluator.default.plain_explain
            return {"Plan": compact_plan(plan["Plan"])}
        candidate = next(
            (
                item
                for item in self.evaluator.candidates
                if item.candidate_id == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError("candidate_id was not issued by the server")
        plan = candidate.plain_explain or candidate.measured_explain_analyze
        if plan is None:
            raise ValueError("candidate has no PostgreSQL plan")
        return {"Plan": compact_plan(plan["Plan"])}

    def execute(self, name: str, arguments: Any) -> tuple[Any, bool]:
        if not isinstance(arguments, dict):
            if name == ToolName.EVALUATE_CANDIDATE:
                return self.evaluator.evaluate(arguments).feedback(), False
            return {"error": "tool arguments must be an object"}, False
        try:
            if name == ToolName.EVALUATE_CANDIDATE:
                return self.evaluator.evaluate(
                    arguments.get("action")
                ).feedback(), False
            if name == ToolName.KEEP_DEFAULT:
                return self.evaluator.keep_default(), True
            if name == ToolName.FINISH:
                if not self.evaluator.candidates:
                    raise RuntimeError(
                        "finish requires a candidate; use keep_default to "
                        "keep PostgreSQL's plan"
                    )
                return {"status": ToolResultStatus.FINISHED.value}, True
            methods = {
                ToolName.DESCRIBE_TABLE.value: self.describe_table,
                ToolName.LIST_INDEXES.value: self.list_indexes,
                ToolName.GET_COLUMN_STATS.value: self.get_column_stats,
                ToolName.GET_RELATION_SIZE.value: self.get_relation_size,
                ToolName.GET_EXTENDED_STATS.value: self.get_extended_stats,
                ToolName.GET_PLAN.value: self.get_plan,
            }
            method = methods.get(name)
            if method is None:
                raise ValueError("unknown tool")
            return method(arguments), False
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            return {"error": str(error)}, False
