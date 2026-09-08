"""Known initial context, kept typed until the message/evidence boundary."""

from typing import Literal

from pydantic import BaseModel, Field

from qorl.agent.presentation import PlanView
from qorl.postgres.schemas import (
    PlannerSettings,
    PostgresResourceLimits,
    WorkerAllocation,
)
from qorl.taskset.schemas import Relation


class DecisionTurns(BaseModel):
    candidate_evaluations: int
    finish_or_keep_default: int = 1


class TurnBudget(BaseModel):
    total_model_turns: int
    maximum_inspection_turns: int
    reserved_final_turns: int
    reserved_for: DecisionTurns


class ContextBudget(BaseModel):
    maximum_tokens: int
    reserved_for_next_completion: int


class RemainingTurnBudget(BaseModel):
    current_turn: int
    turns_remaining: int
    unrestricted_turns_remaining: int
    reserved_final_turns: int


class ResourceLimits(BaseModel):
    worker: WorkerAllocation | None
    worker_source: Literal["startup_verified_container_limits", "unavailable"]
    postgres: PostgresResourceLimits
    work_mem_unit: str = (
        "PostgreSQL setting; bare numbers are kB. Per operation, not query total."
    )


class AgentObservation(BaseModel):
    task_id: str
    objective: str = "minimize measured warm-cache execution time"
    sql: str
    relations: list[Relation]
    join_edges: list[str]
    indexes: dict[str, list[str]]
    planner_settings: PlannerSettings
    default_plan: PlanView
    resource_limits: ResourceLimits
    default_median_execution_time_ms: float | None
    candidate_attempts: int
    candidate_timeout_ms: int
    candidate_feedback_warmups: int = 0
    candidate_feedback_measurements: int = 0
    execution_mode: Literal["plan_only", "final_only", "execution_feedback"] = (
        "plan_only"
    )
    turn_budget: TurnBudget
    context_budget: ContextBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
