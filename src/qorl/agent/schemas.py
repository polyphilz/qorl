"""Conversation budgets, independent of the model provider."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.agent.types import StopReason
from qorl.measure.schemas import SelectionState, SelectionStatus
from qorl.model.schemas import (
    GenerationRequest,
    GenerationResponse,
    JsonObject,
    Message,
    ModelResponseFailure,
    TokenUsage,
    ToolDefinition,
)
from qorl.postgres.schemas import (
    PlannerSettings,
    PostgresResourceLimits,
    WorkerAllocation,
)
from qorl.taskset.schemas import Relation


class PlanNode(BaseModel):
    node_id: str
    parent_id: str | None
    child_ids: list[str]
    leaf_aliases: list[str]
    estimates: JsonObject
    observed: JsonObject | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    omitted_fields: list[str] = Field(default_factory=list)


class PlanView(BaseModel):
    source: str = "plain_explain"
    cost_unit: str = "planner_units_not_milliseconds"
    row_kind: str = "estimated"
    root_node_id: str
    nodes: list[PlanNode]
    omitted_nodes: int


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


class ExecutionObservation(BaseModel):
    status: Literal["collecting", "completed", "timed_out"]
    source_id: str
    reused: bool
    new_executions: int
    warmup_count: int
    measurement_count: int
    median_execution_time_ms: float | None
    preliminary_ratio_to_initial_default: float | None
    ratio_meaning: str = "Preliminary initial-default median / feedback median; not final paired speedup or reward. Warmups excluded."
    timeout_ms: int | None
    displayed_sample_phase: Literal["warmup", "measurement"] | None
    displayed_sample_index: int | None
    displayed_sample_execution_time_ms: float | None
    plan: PlanView | None


class CandidateSummary(BaseModel):
    candidate_id: str
    action_valid: bool
    constraints_satisfied: bool
    selection_eligible: bool
    timeout_phase: Literal["planning", "execution"] | None
    timeout_ms: int | None
    feedback_source_id: str | None
    feedback_median_execution_time_ms: float | None
    preliminary_ratio_to_initial_default: float | None


class CandidateHistory(BaseModel):
    candidates: list[CandidateSummary]
    selectable_candidate_ids: list[str]
    attempts_remaining: int
    selection_status: SelectionStatus
    selected_candidate_id: str | None


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_attempts: int = Field(ge=1)
    maximum_model_turns: int = Field(ge=1)
    inspection_turns_per_alias: int = Field(ge=0)


class ToolEvent(BaseModel):
    """One attempted tool call and the exact result added to the conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: int
    tool_call_id: str
    name: str
    result: JsonObject


class AgentTrace(BaseModel):
    """Conversation evidence, including provider continuation and partial progress."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    agent_interface_version: int
    seed: int | None
    stop_reason: StopReason | None = None
    selection: SelectionState = Field(default_factory=SelectionState)
    initial_observation: AgentObservation
    tools: list[ToolDefinition]
    tools_sha256: str
    transcript: list[Message]
    model_responses: list[GenerationResponse] = Field(
        default_factory=list[GenerationResponse]
    )
    model_requests: list[GenerationRequest] = Field(
        default_factory=list[GenerationRequest]
    )
    tool_events: list[ToolEvent] = Field(default_factory=list[ToolEvent])
    usage: TokenUsage = TokenUsage()
    prompt_tokens: int | None = None
    model_failures: list[ModelResponseFailure] = Field(
        default_factory=list[ModelResponseFailure], exclude_if=lambda value: not value
    )
