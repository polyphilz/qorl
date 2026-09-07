"""Independent evaluation settings, saved conversations, and aggregate evidence."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.agent.schemas import AgentTrace
from qorl.agent.types import StopReason
from qorl.measure.schemas import OutcomeKind, RolloutRecord, RunStatus
from qorl.model.schemas import LocalServerIdentity, ModelSettings, TokenUsage
from qorl.taskset.schemas import Task, TaskRole, TaskSelection
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollouts_per_task: int = Field(ge=1)


class PerformanceSummary(BaseModel):
    """Unclipped latency statistics; failed and timed-out rollouts have no speedup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_count: int
    scored_rollout_count: int
    failure_count: int
    timeout_count: int
    no_valid_candidate_count: int
    geometric_mean_speedup: float | None
    candidate_workload_time_ms: float
    default_workload_time_ms: float
    total_workload_speedup: float | None
    regression_count: int


@dataclass(frozen=True)
class EvaluationItem:
    """One selected task and independent conversation index, starting at zero."""

    task: Task
    rollout_index: int


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
