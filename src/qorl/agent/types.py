from enum import StrEnum
from typing import Protocol

from qorl.db.fixture import DatabaseFixture
from qorl.measure.rollout import QueryExecutor

# This internal-looking key is already model-visible and therefore wire-stable.
TURN_BUDGET_FIELD = "_turn_budget"


class ToolName(StrEnum):
    DESCRIBE_TABLE = "describe_table"
    LIST_INDEXES = "list_indexes"
    GET_COLUMN_STATS = "get_column_stats"
    GET_RELATION_SIZE = "get_relation_size"
    GET_EXTENDED_STATS = "get_extended_stats"
    GET_PLAN = "get_plan"
    EVALUATE_CANDIDATE = "evaluate_candidate"
    KEEP_DEFAULT = "keep_default"
    FINISH = "finish"


class StopReason(StrEnum):
    MODEL_TURN_LIMIT = "model_turn_limit"
    MODEL_FINISH = "model_finish"
    MODEL_KEEP_DEFAULT = "model_keep_default"
    CONTEXT_BUDGET = "context_budget"
    MISSING_TOKEN_USAGE = "missing_token_usage"


TERMINAL_STOP_REASON = {
    ToolName.FINISH: StopReason.MODEL_FINISH,
    ToolName.KEEP_DEFAULT: StopReason.MODEL_KEEP_DEFAULT,
}


class InspectionExecutor(QueryExecutor, Protocol):
    fixture: DatabaseFixture

    def settings(self, names: set[str]) -> dict[str, str]: ...

    def admin_sql(self, sql: str) -> str: ...
