from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from qorl.postgres.schemas import ExplainResult
from qorl.taskset.schemas import BenchmarkId
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest

MIN_SCORE = 0.1
MAX_SCORE = 10.0
INVALID_ATTEMPT_PENALTY = 0.10
DUPLICATE_ATTEMPT_PENALTY = 0.05
TIMEOUT_ATTEMPT_PENALTY = 0.10
NO_VALID_CANDIDATE_REWARD = -3.0
MIN_CALIBRATION_RUNS = 2


class RolloutMeasurementSettings(BaseModel):
    """Initial baseline and final paired timings, with per-statement timeouts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    default_warmups: int = Field(ge=0)
    default_measurements: int = Field(ge=1)
    paired_warmups: int = Field(ge=0)
    paired_measurements: int = Field(ge=1)
    default_timeout_seconds: float = Field(gt=0)
    candidate_timeout_floor_seconds: float = Field(gt=0)
    candidate_timeout_multiplier: float = Field(gt=0)


class CalibrationSettings(BaseModel):
    """Adaptive warmups followed by repeated default-query measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_warmup_runs: int = Field(ge=MIN_CALIBRATION_RUNS)
    num_trials: int = Field(ge=MIN_CALIBRATION_RUNS)
    default_timeout_seconds: float = Field(gt=0)


class RunStatus(StrEnum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    PASSED = "passed"


class MeasurementProtocolId(StrEnum):
    RIGOROUS_EVALUATION_V1 = "rigorous-evaluation-v1"
    RL_TRAINING_V1 = "rl-training-v1"
    RL_TRAINING_V2 = "rl-training-v2"


class FinalStatus(StrEnum):
    COMPLETED = "completed"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    CANDIDATE_TIMEOUT = "candidate_timeout"


class Decision(StrEnum):
    KEEP_DEFAULT = "keep_default"
    CANDIDATE = "candidate"


class ScoreSource(StrEnum):
    EXPLICIT_KEEP_DEFAULT = "explicit_keep_default"
    DEFAULT_FINGERPRINT = "default_fingerprint"
    INTERLEAVED_MEASUREMENT = "interleaved_measurement"


class MeasurementStatus(StrEnum):
    NOT_MEASURED = "not_measured"
    MEASURED = "measured"


class ToolResultStatus(StrEnum):
    FINISHED = "finished"
    KEPT_DEFAULT = "kept_default"


class OutcomeKind(StrEnum):
    KEPT_DEFAULT = "kept_default"
    DEFAULT_DUPLICATE = "default_duplicate"
    MEASURED = "measured"
    TIMED_OUT = "timed_out"
    NO_VALID_CANDIDATE = "no_valid_candidate"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)


class Measurement(Record):
    execution_time_ms: float
    planning_time_ms: float
    plan_sha256: str
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0


class QueryObservation(Measurement):
    run: int


class CalibrationSummary(BaseModel):
    """Timing variation and plan identity across measured trials, excluding warmups."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    measurement_count: int
    warmup_stable: bool
    median_execution_time_ms: float
    mean_execution_time_ms: float
    sample_standard_deviation_ms: float
    coefficient_of_variation: float | None
    minimum_execution_time_ms: float
    maximum_execution_time_ms: float
    distinct_plan_count: int
    plan_sha256s: list[str]


class CalibrationTaskRecord(BaseModel):
    """One task's actual executions, including partial evidence on failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[2] = 2
    task_id: str
    template_id: str
    completed_at_utc: str
    statement_timeout_ms: int
    worker: WorkerManifest
    warmups: list[QueryObservation]
    measurements: list[QueryObservation]


class CalibratedTask(CalibrationTaskRecord):
    status: Literal[RunStatus.COMPLETED] = RunStatus.COMPLETED
    summary: CalibrationSummary
    representative_explain_analyze: dict[str, JsonValue]


class FailedCalibrationTask(CalibrationTaskRecord):
    status: Literal[RunStatus.FAILED] = RunStatus.FAILED
    error_type: str
    error: str


type CalibrationTaskResult = Annotated[
    CalibratedTask | FailedCalibrationTask, Field(discriminator="status")
]


class CalibrationReport(BaseModel):
    """Progress and applied configuration for one calibration stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal[3] = 3
    benchmark_id: BenchmarkId
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None = None
    measurement: CalibrationSettings
    statement_timeout_ms: int
    minimum_warmup_runs: int
    buffer_stability_relative_tolerance: float
    plan_fingerprint_version: int
    worker_pool: PoolManifest | None = None
    task_count: int
    completed_task_count: int = 0
    failed_task_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class QueryRun:
    is_warmup: bool
    observation: QueryObservation
    explain: ExplainResult


class CandidateTimeout(Record):
    timeout_ms: int
    source: str
    manifest_id: str | None
    calibrated_default_median_ms: float | None


class Baseline(Record):
    measurement_protocol_id: MeasurementProtocolId | None = None
    plan_sha256: str
    structural_plan_sha256: str | None = None
    timing_reuse_key: str | None = None
    plan_fingerprint_version: int | None = None
    plain_explain: dict[str, Any]
    median_execution_time_ms: float | None
    warmup: Measurement | None = None
    measurements: list[Measurement] = []
    candidate_timeout: CandidateTimeout | None = None
    compact_plan: dict[str, Any]


class Candidate(Record):
    """One attempt's evidence; duplicate_of denotes execution reuse, not novelty.

    structural_duplicate_of identifies a matching physical plan. plan_sha256
    retains estimates; timing_reuse_key also binds the per-query overrides.
    """

    candidate_id: str
    action: Any
    action_valid: bool
    constraints_satisfied: bool
    compiled_hint: str
    duplicate_of: str | None
    plan_sha256: str | None
    structural_plan_sha256: str | None = None
    structural_duplicate_of: str | None = None
    timing_reuse_key: str | None = None
    plan_fingerprint_version: int | None = None
    provisional_measurements: list[Measurement] = []
    provisional_speedup: float | None
    errors_or_diagnostics: list[str] = []
    pg_hint_plan: dict[str, str] | None
    attempts_remaining: int
    plain_explain: dict[str, Any] | None = None
    compact_plan: dict[str, Any] | None = None
    warmup: Measurement | None = None
    measured_explain_analyze: dict[str, Any] | None = None
    provisional_median_execution_time_ms: float | None = None
    execution_timed_out: bool = False
    timeout_ms: int | None = None
    measurement_status: MeasurementStatus | None = None

    def feedback(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_valid": self.action_valid,
            "constraints_satisfied": self.constraints_satisfied,
            "compiled_hint": self.compiled_hint,
            "duplicate_of": self.duplicate_of,
            "plan_sha256": self.plan_sha256,
            "structural_plan_sha256": self.structural_plan_sha256,
            "structural_duplicate_of": self.structural_duplicate_of,
            "compact_plan": self.compact_plan,
            "planning_time_ms": [
                item.planning_time_ms for item in self.provisional_measurements
            ],
            "execution_time_ms": [
                item.execution_time_ms for item in self.provisional_measurements
            ],
            "provisional_speedup": self.provisional_speedup,
            "execution_timed_out": self.execution_timed_out,
            "timeout_ms": self.timeout_ms,
            "errors_or_diagnostics": self.errors_or_diagnostics,
            "attempts_remaining": self.attempts_remaining,
        }

    @property
    def structurally_novel(self) -> bool:
        """Novelty requires validated structure, not merely a different full hash."""
        return (
            self.action_valid
            and self.constraints_satisfied
            and self.structural_plan_sha256 is not None
            and self.structural_duplicate_of is None
        )


class Outcome(Record):
    measurement_protocol_id: MeasurementProtocolId
    status: FinalStatus
    winning_candidate_id: str | None
    score: float
    trajectory_reward: float
    invalid_attempt_count: int
    duplicate_attempt_count: int
    timeout_attempt_count: int
    decision: Decision | None = None
    winning_plan_sha256: str | None = None
    score_source: ScoreSource | None = None
    pair_orders: list[list[str]] = []
    candidate_measurements: list[Measurement] = []
    default_measurements: list[Measurement] = []
    candidate_median_execution_time_ms: float | None = None
    default_median_execution_time_ms: float | None = None
    timeout_ms: int | None = None

    @property
    def kind(self) -> OutcomeKind:
        if self.status == FinalStatus.NO_VALID_CANDIDATE:
            return OutcomeKind.NO_VALID_CANDIDATE
        if self.status == FinalStatus.CANDIDATE_TIMEOUT:
            return OutcomeKind.TIMED_OUT
        if self.decision == Decision.KEEP_DEFAULT:
            return OutcomeKind.KEPT_DEFAULT
        if self.score_source == ScoreSource.DEFAULT_FINGERPRINT:
            return OutcomeKind.DEFAULT_DUPLICATE
        return OutcomeKind.MEASURED


def score(default_median_ms: float, candidate_median_ms: float) -> float:
    return min(MAX_SCORE, max(MIN_SCORE, default_median_ms / candidate_median_ms))


def measured_reward(
    score_value: float,
    invalid_attempts: int,
    duplicate_attempts: int,
    timeout_attempts: int = 0,
    *,
    include_quality: bool = True,
) -> float:
    quality = math.log(score_value) if include_quality else 0.0
    return (
        quality
        - INVALID_ATTEMPT_PENALTY * invalid_attempts
        - DUPLICATE_ATTEMPT_PENALTY * duplicate_attempts
        - TIMEOUT_ATTEMPT_PENALTY * timeout_attempts
    )
