"""Independent evaluation settings, saved conversations, and aggregate evidence."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.agent.schemas import AgentTrace
from qorl.agent.types import StopReason
from qorl.measure.schemas import ExecutionCounts, OutcomeKind, RolloutRecord, RunStatus
from qorl.model.schemas import LocalServerIdentity, ModelSettings, TokenUsage
from qorl.taskset.schemas import Task, TaskRole, TaskSelection
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollouts_per_task: int = Field(ge=1)
    selection_policy: Literal["model", "best_feedback"] = "model"


class PerformanceSummary(BaseModel):
    """Unclipped latency statistics; failed and timed-out rollouts have no speedup.

    Selected positions are one-based over all issued attempts, including invalid
    attempts, not speed ranks. An earlier choice precedes the last issued attempt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_count: int
    scored_rollout_count: int
    failure_count: int
    timeout_count: int
    no_valid_candidate_count: int
    selection_failure_count: int = 0
    selected_candidate_positions: dict[int, int] = Field(default_factory=dict[int, int])
    earlier_candidate_selection_count: int = 0
    rejected_selection_count: int = 0
    geometric_mean_speedup: float | None
    candidate_workload_time_ms: float
    default_workload_time_ms: float
    total_workload_speedup: float | None
    regression_count: int
    execution_counts: ExecutionCounts = Field(default_factory=ExecutionCounts)
    execution_accounting_missing: int = 0


@dataclass(frozen=True)
class EvaluationItem:
    """One selected task and independent conversation index, starting at zero."""

    task: Task
    rollout_index: int


class FeedbackSelection(BaseModel):
    """A rule chosen before final timing, alongside the unchanged model outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_speedup: float = Field(default=1.05, ge=1.05, le=1.05)
    model_selected_candidate_id: str | None
    selected_candidate_id: str
    measurement_seed: int
    reused_model_result: bool
    rollout: RolloutRecord


class FeedbackSelectionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_speedup: float = Field(default=1.05, ge=1.05, le=1.05)
    applied_rollout_count: int
    changed_selection_count: int
    default_selection_count: int
    reused_model_result_count: int
    performance: PerformanceSummary
    additional_execution_counts: ExecutionCounts


class EvaluationRollout(BaseModel):
    """A conversation and its measurement evidence, including partial failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    rollout_index: int = Field(ge=0)
    model_seed: int
    measurement_seed: int
    started_at_utc: str
    completed_at_utc: str
    worker: WorkerManifest | None
    rollout: RolloutRecord
    trace: AgentTrace | None
    feedback_selection: FeedbackSelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class EvaluationSummary(BaseModel):
    """Plan rates use all saved rollouts, including failures; latency excludes unknowns.

    Validity and novelty count any validated attempt, even in partial failure
    evidence. Distinct novelty counts task/structural-plan pairs across rollouts.
    Missing records are reported separately, never classified as policy failures.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recorded_rollout_count: int
    unrecorded_rollout_count: int
    recorded_task_count: int
    valid_plan_rollout_count: int
    novel_plan_rollout_count: int
    distinct_novel_plan_count: int
    valid_plan_rate: float | None
    novel_plan_rate: float | None
    distinct_novel_plan_yield: float | None
    candidate_attempt_count: int
    outcome_counts: dict[OutcomeKind, int]
    stop_reason_counts: dict[StopReason, int]
    usage: TokenUsage
    performance: PerformanceSummary
    feedback_selection: FeedbackSelectionSummary | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class EvaluationReport(BaseModel):
    """One fresh evaluation invocation; model includes the explicitly selected adapter."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    status: RunStatus
    split: TaskRole
    selection: TaskSelection
    model: ModelSettings
    seed: int
    settings: EvaluationSettings
    plan_fingerprint_version: int
    started_at_utc: str
    completed_at_utc: str | None = None
    elapsed_seconds: float | None = None
    local_server: LocalServerIdentity | None = None
    worker_pool: PoolManifest | None = None
    summary: EvaluationSummary
    error: str | None = None
