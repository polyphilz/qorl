"""Settings for independent live evaluation rollouts."""

from pydantic import BaseModel, ConfigDict, Field


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
