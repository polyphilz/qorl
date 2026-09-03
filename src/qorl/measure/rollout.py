from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any

from qorl.db.exceptions import QueryTimeout, WorkerError
from qorl.db.worker import ExplainResult
from qorl.measure.protocols import QueryExecutor, SqlSource
from qorl.measure.schemas import (
    MIN_SCORE,
    NO_VALID_CANDIDATE_REWARD,
    Baseline,
    Candidate,
    CandidateTimeout,
    Decision,
    FinalStatus,
    Measurement,
    MeasurementProtocolId,
    Outcome,
    ScoreSource,
    ToolResultStatus,
    measured_reward,
    score,
)
from qorl.plans.catalog import TaskCatalog
from qorl.plans.exceptions import ActionError
from qorl.plans.fingerprint import plan_sha256
from qorl.plans.schemas import PlanAction
from qorl.plans.verify import Verification, compact_plan, hint_status, verify_action
from qorl.workload.timeouts import GLOBAL_TIMEOUT_MS, TaskTimeout, task_timeout_ms

DEFAULT_MEASUREMENTS = 3
FINAL_PAIRS = 5
MAX_CANDIDATES = 5


@dataclass(frozen=True)
class MeasurementProtocol:
    protocol_id: MeasurementProtocolId
    default_warmup_runs: int
    default_measurement_runs: int
    candidate_warmup_runs: int
    candidate_measurement_runs: int
    final_warmup_pairs: int
    final_randomized_pairs: int

    @property
    def max_explain_analyze_executions(self) -> int:
        return self.max_explain_analyze_executions_for(MAX_CANDIDATES)

    def max_explain_analyze_executions_for(self, candidate_attempts: int) -> int:
        return (
            self.default_warmup_runs
            + self.default_measurement_runs
            + candidate_attempts
            * (self.candidate_warmup_runs + self.candidate_measurement_runs)
            + 2 * self.final_warmup_pairs
            + 2 * self.final_randomized_pairs
        )

    def manifest(
        self, candidate_attempts: int = MAX_CANDIDATES
    ) -> dict[str, int | str]:
        return {
            "id": self.protocol_id.value,
            "candidate_attempts": candidate_attempts,
            "default_warmup_runs": self.default_warmup_runs,
            "default_measurement_runs": self.default_measurement_runs,
            "novel_candidate_warmup_runs": self.candidate_warmup_runs,
            "novel_candidate_measurement_runs": self.candidate_measurement_runs,
            "final_warmup_runs_per_plan": self.final_warmup_pairs,
            "final_randomized_pair_count": self.final_randomized_pairs,
            "max_explain_analyze_executions": (
                self.max_explain_analyze_executions_for(candidate_attempts)
            ),
        }


RIGOROUS_EVALUATION_PROTOCOL_V1 = MeasurementProtocol(
    protocol_id=MeasurementProtocolId.RIGOROUS_EVALUATION_V1,
    default_warmup_runs=1,
    default_measurement_runs=DEFAULT_MEASUREMENTS,
    candidate_warmup_runs=1,
    candidate_measurement_runs=1,
    final_warmup_pairs=1,
    final_randomized_pairs=FINAL_PAIRS,
)


def training_protocol(protocol_id: MeasurementProtocolId) -> MeasurementProtocol:
    if protocol_id not in {
        MeasurementProtocolId.RL_TRAINING_V1,
        MeasurementProtocolId.RL_TRAINING_V2,
    }:
        raise ValueError(f"not a training protocol: {protocol_id}")
    return MeasurementProtocol(
        protocol_id=protocol_id,
        default_warmup_runs=1,
        default_measurement_runs=1,
        candidate_warmup_runs=0,
        candidate_measurement_runs=1,
        final_warmup_pairs=0,
        final_randomized_pairs=3,
    )


@dataclass(frozen=True)
class PlanTiming:
    candidate_id: str
    measurements: list[Measurement]
    median_execution_time_ms: float | None


def measured(result: ExplainResult) -> Measurement:
    plan = result.document["Plan"]
    return Measurement(
        execution_time_ms=result.document["Execution Time"],
        planning_time_ms=result.document["Planning Time"],
        plan_sha256=plan_sha256(plan),
    )


class RolloutEvaluator[ExecutorT: QueryExecutor]:
    def __init__(
        self,
        worker: ExecutorT,
        task_set: SqlSource,
        task: dict[str, Any],
        *,
        global_timeout_ms: int = GLOBAL_TIMEOUT_MS,
        measurement_protocol: MeasurementProtocol = (RIGOROUS_EVALUATION_PROTOCOL_V1),
        calibrated_timeout: TaskTimeout | None = None,
        timeout_manifest_id: str | None = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self._worker = worker
        self.task = task
        self.sql = task_set.load_sql(task)
        self.global_timeout_ms = global_timeout_ms
        self.measurement_protocol = measurement_protocol
        self.catalog = TaskCatalog.from_task(task, worker.task_indexes(task))
        self.default: Baseline | None = None
        self.candidates: list[Candidate] = []
        self.by_fingerprint: dict[str, PlanTiming] = {}
        self.kept_default = False
        self.calibrated_timeout = calibrated_timeout
        self.timeout_manifest_id = timeout_manifest_id
        self.max_candidates = max_candidates
        self.timeout_ms = (
            calibrated_timeout.timeout_ms
            if calibrated_timeout is not None
            else global_timeout_ms
        )

    @property
    def worker(self) -> ExecutorT:
        return self._worker

    def start(self) -> Baseline:
        baseline_timeout_ms = (
            self.timeout_ms
            if self.calibrated_timeout is not None
            else self.global_timeout_ms
        )
        plain = self.worker.explain(self.sql, baseline_timeout_ms)
        default_plan_sha256 = plan_sha256(plain.document["Plan"])
        if (
            self.calibrated_timeout is not None
            and default_plan_sha256 not in self.calibrated_timeout.plan_sha256s
        ):
            raise RuntimeError(
                "default plan differs from calibrated plan: "
                f"task={self.task['task_id']} "
                f"actual={default_plan_sha256}"
            )
        warmups = [
            self.worker.explain(self.sql, baseline_timeout_ms, analyze=True)
            for _ in range(self.measurement_protocol.default_warmup_runs)
        ]
        measurements = [
            measured(self.worker.explain(self.sql, baseline_timeout_ms, analyze=True))
            for _ in range(self.measurement_protocol.default_measurement_runs)
        ]
        median_ms = statistics.median(item.execution_time_ms for item in measurements)
        if self.calibrated_timeout is None:
            self.timeout_ms = task_timeout_ms(median_ms, self.global_timeout_ms)
        self.default = Baseline(
            measurement_protocol_id=self.measurement_protocol.protocol_id,
            plan_sha256=default_plan_sha256,
            plain_explain=plain.document,
            warmup=measured(warmups[-1]) if warmups else None,
            measurements=measurements,
            median_execution_time_ms=median_ms,
            candidate_timeout=CandidateTimeout(
                timeout_ms=self.timeout_ms,
                source=(
                    "calibrated"
                    if self.calibrated_timeout is not None
                    else "live_default"
                ),
                manifest_id=self.timeout_manifest_id,
                calibrated_default_median_ms=(
                    self.calibrated_timeout.calibrated_default_ms
                    if self.calibrated_timeout is not None
                    else None
                ),
            ),
            compact_plan=compact_plan(
                (warmups[-1] if warmups else plain).document["Plan"]
            ),
        )
        self.by_fingerprint[self.default.plan_sha256] = PlanTiming(
            candidate_id="default",
            measurements=measurements,
            median_execution_time_ms=median_ms,
        )
        return self.default

    def invalid_candidate(
        self,
        candidate_id: str,
        action: Any,
        error: str,
        *,
        hint: str = "",
        action_valid: bool = False,
        pg_hint_plan: dict[str, str] | None = None,
        execution_timed_out: bool = False,
    ) -> Candidate:
        result = Candidate(
            candidate_id=candidate_id,
            action=action,
            action_valid=action_valid,
            constraints_satisfied=False,
            compiled_hint=hint,
            duplicate_of=None,
            plan_sha256=None,
            provisional_measurements=[],
            provisional_speedup=MIN_SCORE if execution_timed_out else 0.0,
            execution_timed_out=execution_timed_out,
            timeout_ms=self.timeout_ms if execution_timed_out else None,
            errors_or_diagnostics=[error],
            pg_hint_plan=pg_hint_plan,
            attempts_remaining=self.max_candidates - len(self.candidates) - 1,
        )
        self.candidates.append(result)
        return result

    def evaluate(self, raw_action: Any) -> Candidate:
        if self.default is None:
            raise RuntimeError("rollout baseline has not been started")
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        if len(self.candidates) >= self.max_candidates:
            raise RuntimeError("rollout candidate budget is exhausted")
        candidate_id = f"candidate-{len(self.candidates) + 1:02d}"
        try:
            plan_action = PlanAction.from_raw(raw_action, self.catalog)
            action = plan_action.to_wire()
            hint = plan_action.compile()
        except ActionError as error:
            return self.invalid_candidate(candidate_id, raw_action, str(error))

        try:
            plain = self.worker.explain(self.sql, self.timeout_ms, hint=hint)
        except QueryTimeout as error:
            return self.invalid_candidate(
                candidate_id,
                action,
                str(error),
                hint=hint,
                action_valid=True,
                execution_timed_out=True,
            )
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        verification = verify_action(
            action, plain.document["Plan"], plain.hint_diagnostics
        )
        pg_hint_plan = hint_status(plain.hint_diagnostics)
        if not verification.valid:
            return self.invalid_candidate(
                candidate_id,
                action,
                "; ".join(verification.errors),
                hint=hint,
                action_valid=True,
                pg_hint_plan=pg_hint_plan,
            )

        fingerprint = plan_sha256(plain.document["Plan"])
        duplicate = self.by_fingerprint.get(fingerprint)
        if duplicate is not None:
            if duplicate.median_execution_time_ms is None:
                raise RuntimeError("duplicate plan has no timing measurement")
            if self.default.median_execution_time_ms is None:
                raise RuntimeError("default plan has no timing measurement")
            result = Candidate(
                candidate_id=candidate_id,
                action=action,
                action_valid=True,
                constraints_satisfied=True,
                compiled_hint=hint,
                duplicate_of=duplicate.candidate_id,
                plan_sha256=fingerprint,
                plain_explain=plain.document,
                compact_plan=compact_plan(plain.document["Plan"]),
                provisional_measurements=duplicate.measurements,
                provisional_median_execution_time_ms=(
                    duplicate.median_execution_time_ms
                ),
                provisional_speedup=score(
                    self.default.median_execution_time_ms,
                    duplicate.median_execution_time_ms,
                ),
                errors_or_diagnostics=[],
                pg_hint_plan=pg_hint_plan,
                attempts_remaining=self.max_candidates - len(self.candidates) - 1,
            )
            self.candidates.append(result)
            return result

        try:
            warmups = [
                self.checked_execution(action, hint)
                for _ in range(self.measurement_protocol.candidate_warmup_runs)
            ]
            executions = [
                self.checked_execution(action, hint)
                for _ in range(self.measurement_protocol.candidate_measurement_runs)
            ]
        except QueryTimeout as error:
            result = Candidate(
                candidate_id=candidate_id,
                action=action,
                action_valid=True,
                constraints_satisfied=True,
                compiled_hint=hint,
                duplicate_of=None,
                plan_sha256=fingerprint,
                plain_explain=plain.document,
                compact_plan=compact_plan(plain.document["Plan"]),
                provisional_measurements=[],
                provisional_speedup=MIN_SCORE,
                execution_timed_out=True,
                timeout_ms=error.timeout_ms,
                errors_or_diagnostics=[str(error)],
                pg_hint_plan=pg_hint_plan,
                attempts_remaining=self.max_candidates - len(self.candidates) - 1,
            )
            self.candidates.append(result)
            return result
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        observations = [measured(execution) for execution in executions]
        median_ms = statistics.median(
            observation.execution_time_ms for observation in observations
        )
        if self.default.median_execution_time_ms is None:
            raise RuntimeError("default plan has no timing measurement")
        result = Candidate(
            candidate_id=candidate_id,
            action=action,
            action_valid=True,
            constraints_satisfied=True,
            compiled_hint=hint,
            duplicate_of=None,
            plan_sha256=fingerprint,
            plain_explain=plain.document,
            warmup=measured(warmups[-1]) if warmups else None,
            measured_explain_analyze=executions[-1].document,
            compact_plan=compact_plan(executions[-1].document["Plan"]),
            provisional_measurements=observations,
            provisional_median_execution_time_ms=median_ms,
            provisional_speedup=score(self.default.median_execution_time_ms, median_ms),
            execution_timed_out=False,
            timeout_ms=self.timeout_ms,
            errors_or_diagnostics=[],
            pg_hint_plan=pg_hint_plan,
            attempts_remaining=self.max_candidates - len(self.candidates) - 1,
        )
        self.candidates.append(result)
        self.by_fingerprint[fingerprint] = PlanTiming(
            candidate_id=candidate_id,
            measurements=observations,
            median_execution_time_ms=median_ms,
        )
        return result

    def keep_default(self) -> dict[str, str]:
        if self.default is None:
            raise RuntimeError("rollout baseline has not been started")
        if self.candidates:
            raise RuntimeError(
                "keep_default must be selected before submitting a candidate"
            )
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        self.kept_default = True
        return {"status": ToolResultStatus.KEPT_DEFAULT.value}

    def checked_execution(self, action: dict[str, Any], hint: str) -> ExplainResult:
        result = self.worker.explain(self.sql, self.timeout_ms, analyze=True, hint=hint)
        verification: Verification = verify_action(
            action, result.document["Plan"], result.hint_diagnostics
        )
        if not verification.valid:
            raise WorkerError("; ".join(verification.errors))
        return result

    def finish(self, rng: random.Random) -> Outcome:
        if self.default is None:
            raise RuntimeError("rollout baseline has not been started")
        if self.kept_default:
            median_ms = self.default.median_execution_time_ms
            return Outcome(
                measurement_protocol_id=self.measurement_protocol.protocol_id,
                status=FinalStatus.COMPLETED,
                decision=Decision.KEEP_DEFAULT,
                winning_candidate_id="default",
                winning_plan_sha256=self.default.plan_sha256,
                score_source=ScoreSource.EXPLICIT_KEEP_DEFAULT,
                pair_orders=[],
                candidate_measurements=self.default.measurements,
                default_measurements=self.default.measurements,
                candidate_median_execution_time_ms=median_ms,
                default_median_execution_time_ms=median_ms,
                score=1.0,
                trajectory_reward=0.0,
                invalid_attempt_count=0,
                duplicate_attempt_count=0,
                timeout_attempt_count=0,
            )

        valid = [
            candidate
            for candidate in self.candidates
            if candidate.action_valid
            and candidate.constraints_satisfied
            and not candidate.execution_timed_out
        ]
        invalid_count = len(self.candidates) - len(valid)
        timeout_count = sum(
            candidate.execution_timed_out for candidate in self.candidates
        )
        duplicate_count = sum(candidate.duplicate_of is not None for candidate in valid)
        if not valid:
            return Outcome(
                measurement_protocol_id=self.measurement_protocol.protocol_id,
                status=FinalStatus.NO_VALID_CANDIDATE,
                winning_candidate_id=None,
                score=0.0,
                trajectory_reward=NO_VALID_CANDIDATE_REWARD,
                invalid_attempt_count=invalid_count,
                duplicate_attempt_count=duplicate_count,
                timeout_attempt_count=timeout_count,
            )

        winner = min(
            valid,
            key=lambda candidate: (
                candidate.provisional_median_execution_time_ms
                if candidate.provisional_median_execution_time_ms is not None
                else float("inf")
            ),
        )
        if winner.plan_sha256 == self.default.plan_sha256:
            median_ms = self.default.median_execution_time_ms
            return Outcome(
                measurement_protocol_id=self.measurement_protocol.protocol_id,
                status=FinalStatus.COMPLETED,
                decision=Decision.CANDIDATE,
                winning_candidate_id=winner.candidate_id,
                winning_plan_sha256=winner.plan_sha256,
                score_source=ScoreSource.DEFAULT_FINGERPRINT,
                pair_orders=[],
                candidate_measurements=self.default.measurements,
                default_measurements=self.default.measurements,
                candidate_median_execution_time_ms=median_ms,
                default_median_execution_time_ms=median_ms,
                score=1.0,
                trajectory_reward=measured_reward(
                    1.0,
                    invalid_count,
                    duplicate_count,
                    include_quality=False,
                ),
                invalid_attempt_count=invalid_count,
                duplicate_attempt_count=duplicate_count,
                timeout_attempt_count=timeout_count,
            )

        action = winner.action
        hint = winner.compiled_hint
        for _ in range(self.measurement_protocol.final_warmup_pairs):
            try:
                self.checked_execution(action, hint)
            except QueryTimeout:
                return self.final_timeout(
                    winner, invalid_count, duplicate_count, timeout_count
                )
            self.worker.explain(self.sql, self.timeout_ms, analyze=True)

        pair_orders: list[list[str]] = []
        candidate_measurements: list[Measurement] = []
        default_measurements: list[Measurement] = []
        for _ in range(self.measurement_protocol.final_randomized_pairs):
            order = [Decision.CANDIDATE.value, "default"]
            rng.shuffle(order)
            pair_orders.append(order)
            for label in order:
                if label == Decision.CANDIDATE:
                    try:
                        candidate_measurements.append(
                            measured(self.checked_execution(action, hint))
                        )
                    except QueryTimeout:
                        return self.final_timeout(
                            winner,
                            invalid_count,
                            duplicate_count,
                            timeout_count,
                            pair_orders,
                            candidate_measurements,
                            default_measurements,
                        )
                else:
                    default_measurements.append(
                        measured(
                            self.worker.explain(self.sql, self.timeout_ms, analyze=True)
                        )
                    )

        candidate_median = statistics.median(
            item.execution_time_ms for item in candidate_measurements
        )
        default_median = statistics.median(
            item.execution_time_ms for item in default_measurements
        )
        final_score = score(default_median, candidate_median)
        return Outcome(
            measurement_protocol_id=self.measurement_protocol.protocol_id,
            status=FinalStatus.COMPLETED,
            decision=Decision.CANDIDATE,
            winning_candidate_id=winner.candidate_id,
            winning_plan_sha256=winner.plan_sha256,
            score_source=ScoreSource.INTERLEAVED_MEASUREMENT,
            pair_orders=pair_orders,
            candidate_measurements=candidate_measurements,
            default_measurements=default_measurements,
            candidate_median_execution_time_ms=candidate_median,
            default_median_execution_time_ms=default_median,
            score=final_score,
            trajectory_reward=measured_reward(
                final_score, invalid_count, duplicate_count
            ),
            invalid_attempt_count=invalid_count,
            duplicate_attempt_count=duplicate_count,
            timeout_attempt_count=timeout_count,
        )

    def final_timeout(
        self,
        winner: Candidate,
        invalid_count: int,
        duplicate_count: int,
        timeout_count: int,
        pair_orders: list[list[str]] | None = None,
        candidate_measurements: list[Measurement] | None = None,
        default_measurements: list[Measurement] | None = None,
    ) -> Outcome:
        final_score = MIN_SCORE
        return Outcome(
            measurement_protocol_id=self.measurement_protocol.protocol_id,
            status=FinalStatus.CANDIDATE_TIMEOUT,
            winning_candidate_id=winner.candidate_id,
            winning_plan_sha256=winner.plan_sha256,
            pair_orders=pair_orders or [],
            candidate_measurements=candidate_measurements or [],
            default_measurements=default_measurements or [],
            timeout_ms=self.timeout_ms,
            score=final_score,
            trajectory_reward=measured_reward(
                final_score, invalid_count, duplicate_count
            ),
            invalid_attempt_count=invalid_count,
            duplicate_attempt_count=duplicate_count,
            timeout_attempt_count=timeout_count + 1,
        )
