from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

MIN_SCORE = 0.1
MAX_SCORE = 10.0
INVALID_ATTEMPT_PENALTY = 0.10
DUPLICATE_ATTEMPT_PENALTY = 0.05
NO_VALID_CANDIDATE_REWARD = -3.0


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


class CandidateOutcome(StrEnum):
    MALFORMED = "malformed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    DUPLICATE = "duplicate"
    MEASURED = "measured"


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


class CandidateTimeout(Record):
    timeout_ms: int
    source: str
    manifest_id: str | None
    calibrated_default_median_ms: float | None


class Baseline(Record):
    measurement_protocol_id: MeasurementProtocolId | None = None
    plan_sha256: str
    plain_explain: dict[str, Any]
    median_execution_time_ms: float | None
    warmup: Measurement | None = None
    measurements: list[Measurement] = []
    candidate_timeout: CandidateTimeout | None = None
    compact_plan: dict[str, Any]


class Candidate(Record):
    candidate_id: str
    action: Any
    action_valid: bool
    constraints_satisfied: bool
    compiled_hint: str
    duplicate_of: str | None
    plan_sha256: str | None
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

    @property
    def outcome(self) -> CandidateOutcome:
        if not self.action_valid:
            return CandidateOutcome.MALFORMED
        if self.execution_timed_out:
            return CandidateOutcome.TIMED_OUT
        if not self.constraints_satisfied:
            return CandidateOutcome.REJECTED
        if self.duplicate_of is not None:
            return CandidateOutcome.DUPLICATE
        return CandidateOutcome.MEASURED

    def feedback(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_valid": self.action_valid,
            "constraints_satisfied": self.constraints_satisfied,
            "compiled_hint": self.compiled_hint,
            "duplicate_of": self.duplicate_of,
            "plan_sha256": self.plan_sha256,
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
    *,
    include_quality: bool = True,
) -> float:
    quality = math.log(score_value) if include_quality else 0.0
    return (
        quality
        - INVALID_ATTEMPT_PENALTY * invalid_attempts
        - DUPLICATE_ATTEMPT_PENALTY * duplicate_attempts
    )
