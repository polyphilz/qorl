from enum import StrEnum
from typing import Protocol

from qorl.measure.protocols import QueryExecutor
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.validation import PlanValidationEvaluator
from qorl.postgres.schemas import PostgresSettings

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
    MODEL_OUTPUT_LIMIT = "model_output_limit"


TERMINAL_STOP_REASON = {
    ToolName.FINISH: StopReason.MODEL_FINISH,
    ToolName.KEEP_DEFAULT: StopReason.MODEL_KEEP_DEFAULT,
}


class InspectionExecutor(QueryExecutor, Protocol):
    @property
    def settings(self) -> PostgresSettings: ...

    def admin_sql(self, sql: str) -> str: ...


type AgentEvaluator = (
    RolloutEvaluator[InspectionExecutor] | PlanValidationEvaluator[InspectionExecutor]
)
