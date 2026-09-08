"""Measure the initial default, validate attempts, then time one selected candidate."""

import random
import statistics
from threading import Event

from pydantic import JsonValue

from qorl.measure.protocols import QueryExecutor, SqlSource
from qorl.measure.schemas import (
    Baseline,
    Candidate,
    DefaultDuplicateOutcome,
    ExecutionCounts,
    ExecutionFeedback,
    KeptDefaultOutcome,
    MeasuredOutcome,
    MeasuredPlan,
    Measurement,
    MeasurementPair,
    MeasurementStatus,
    NoValidCandidateOutcome,
    Outcome,
    PairedMeasurements,
    RolloutFailure,
    RolloutMeasurementSettings,
    RolloutRecord,
    SelectionFailedOutcome,
    SelectionStatus,
    TimedOutOutcome,
)
from qorl.measure.timeouts import candidate_timeout_ms, seconds_to_ms
from qorl.measure.validation import PlanValidationEvaluator
from qorl.plans.fingerprint import plan_sha256
from qorl.plans.schemas import PlanAction
from qorl.plans.verify import verify_action
from qorl.postgres.exceptions import PostgresError, QueryTimeoutError
from qorl.postgres.schemas import ExplainResult
from qorl.taskset.schemas import Task


def measured(result: ExplainResult) -> Measurement:
    """Extract actual timing, buffers and full plan identity from EXPLAIN ANALYZE."""
    plan = result.document["Plan"]
    return Measurement(
        execution_time_ms=result.document["Execution Time"],
        planning_time_ms=result.document["Planning Time"],
        plan_sha256=plan_sha256(plan),
        shared_hit_blocks=plan.get("Shared Hit Blocks", 0),
        shared_read_blocks=plan.get("Shared Read Blocks", 0),
        analyzed_document=result.document,
        hint_diagnostics=result.hint_diagnostics,
    )


class RolloutEvaluator[ExecutorT: QueryExecutor](PlanValidationEvaluator[ExecutorT]):
    """Own initial timing, optional candidate probes, and fresh selected final pairs."""

    def __init__(
        self,
        worker: ExecutorT,
        task_set: SqlSource,
        task: Task,
        *,
        measurement: RolloutMeasurementSettings,
        max_candidates: int,
        cancel: Event | None = None,
    ) -> None:
        super().__init__(
            worker,
            task_set,
            task,
            default_timeout_ms=seconds_to_ms(measurement.default_timeout_seconds),
            max_candidates=max_candidates,
            cancel=cancel,
        )
        self.measurement = measurement
        self.default_timeout_ms = self.timeout_ms
        self.paired = PairedMeasurements()
        self.final: Outcome | None = None
        self.operation = "initial_default_explain"
        self.finalization_started = False
        self.execution_counts = ExecutionCounts()

    def start(self) -> Baseline:
        """Keep every initial execution and derive the uncapped candidate deadline."""
        baseline = super().start()
        warmups: list[Measurement] = []
        measurements: list[Measurement] = []
        # Retain each completed execution, including before a later failure.
        self.default = baseline.model_copy(
            update={"warmups": warmups, "measurements": measurements}
        )
        self.operation = "initial_default_warmup"
        for _ in range(self.measurement.default_warmups):
            self.check_cancelled()
            self.execution_counts.initial_default += 1
            warmups.append(
                measured(
                    self.worker.explain(self.sql, self.default_timeout_ms, analyze=True)
                )
            )
            if warmups[-1].plan_sha256 != baseline.plan_sha256:
                raise PostgresError("default plan changed during initial warmup")
        self.operation = "initial_default_measurement"
        for _ in range(self.measurement.default_measurements):
            self.check_cancelled()
            self.execution_counts.initial_default += 1
            measurements.append(
                measured(
                    self.worker.explain(self.sql, self.default_timeout_ms, analyze=True)
                )
            )
            if measurements[-1].plan_sha256 != baseline.plan_sha256:
                raise PostgresError("default plan changed during initial measurement")
        self.check_cancelled()
        median_ms = statistics.median(item.execution_time_ms for item in measurements)
        if median_ms <= 0:
            raise PostgresError(
                "initial default median execution time must be positive"
            )
        self.timeout_ms = candidate_timeout_ms(median_ms, self.measurement)
        self.default = self.default.model_copy(
            update={
                "median_execution_time_ms": median_ms,
                "candidate_timeout_ms": self.timeout_ms,
            }
        )
        self.operation = "agent"
        return self.default

    def check_attempt(self) -> None:
        """Accept validation attempts only after initial timing and before finalization."""
        if self.finalization_started:
            raise RuntimeError("rollout finalization has already started")
        if self.default is None or self.default.median_execution_time_ms is None:
            raise RuntimeError("rollout baseline has not been measured")
        super().check_attempt()

    def evaluate(self, raw_action: JsonValue) -> Candidate:
        candidate = super().evaluate(raw_action)
        if (
            not candidate.constraints_satisfied
            or not self.measurement.candidate_feedback_measurements
        ):
            return candidate
        baseline = self.default
        if baseline is None:
            raise RuntimeError("rollout baseline has not been measured")
        feedback = ExecutionFeedback()
        if candidate.timing_reuse_key == baseline.timing_reuse_key:
            feedback = ExecutionFeedback(status="completed", source_id="default")
        else:
            source = next(
                (
                    item
                    for item in self.candidates[:-1]
                    if item.timing_reuse_key == candidate.timing_reuse_key
                    and item.execution_feedback is not None
                    and item.execution_feedback.status in ("completed", "timed_out")
                ),
                None,
            )
            if source is not None and source.execution_feedback is not None:
                feedback = ExecutionFeedback(
                    status=source.execution_feedback.status,
                    source_id=source.execution_feedback.source_id
                    or source.candidate_id,
                    timeout_ms=source.execution_feedback.timeout_ms,
                )
                candidate = candidate.model_copy(
                    update={
                        "execution_timed_out": source.execution_timed_out,
                        "timeout_ms": source.timeout_ms,
                    }
                )
        candidate = candidate.model_copy(update={"execution_feedback": feedback})
        self.candidates[-1] = candidate
        if feedback.source_id is not None:
            return candidate
        for phase, count, samples in (
            ("warmup", self.measurement.candidate_feedback_warmups, feedback.warmups),
            (
                "measurement",
                self.measurement.candidate_feedback_measurements,
                feedback.measurements,
            ),
        ):
            self.operation = f"candidate_feedback_{phase}"
            for _ in range(count):
                self.check_cancelled()
                self.execution_counts.candidate_feedback += 1
                try:
                    result = self.worker.explain(
                        self.sql,
                        self.timeout_ms,
                        analyze=True,
                        hint=candidate.compiled_hint,
                    )
                except QueryTimeoutError as error:
                    candidate = candidate.model_copy(
                        update={
                            "execution_feedback": feedback.model_copy(
                                update={
                                    "status": "timed_out",
                                    "timeout_ms": error.timeout_ms,
                                }
                            )
                        }
                    )
                    self.candidates[-1] = candidate
                    self.mark_timeout(candidate, error)
                    self.check_cancelled()
                    self.operation = "agent"
                    return self.candidates[-1]
                # Lists belong to the already-recorded attempt: cancellation and drift
                # must not erase the execution that just completed.
                samples.append(measured(result))
                self.check_cancelled()
                self.verify_execution(candidate, result, samples[-1])
        if (
            statistics.median(
                sample.execution_time_ms for sample in feedback.measurements
            )
            <= 0
        ):
            raise PostgresError(
                "candidate feedback median execution time must be positive"
            )
        candidate = candidate.model_copy(
            update={
                "execution_feedback": feedback.model_copy(
                    update={"status": "completed"}
                )
            }
        )
        self.candidates[-1] = candidate
        self.operation = "agent"
        return candidate

    def verify_execution(
        self, candidate: Candidate, result: ExplainResult, observation: Measurement
    ) -> None:
        verification = verify_action(
            PlanAction.from_raw(candidate.action, self.catalog).to_wire(),
            result.document["Plan"],
            result.hint_diagnostics,
        )
        if not verification.valid:
            raise PostgresError(
                "candidate hints changed during measurement: "
                + "; ".join(verification.errors)
            )
        if observation.plan_sha256 != candidate.plan_sha256:
            raise PostgresError("candidate plan changed during measurement")

    def finish(
        self, rng: random.Random, *, selected_candidate_id: str | None = None
    ) -> Outcome:
        """Resolve one outcome; default errors propagate as unscored failures."""
        self.check_cancelled()
        baseline = self.default
        if baseline is None or baseline.median_execution_time_ms is None:
            raise RuntimeError("rollout baseline has not been measured")
        if self.finalization_started:
            raise RuntimeError("rollout finalization has already started")
        if self.kept_default and selected_candidate_id is not None:
            raise ValueError("keep_default cannot select a candidate")
        if self.selection.status in (SelectionStatus.REJECTED, SelectionStatus.FAILED):
            self.selection.status = SelectionStatus.FAILED
            self.finalization_started = True
            self.final = SelectionFailedOutcome()
            return self.final
        selected = self.select(
            self.selection.selected_candidate_id
            if selected_candidate_id is None
            else selected_candidate_id
        )
        self.finalization_started = True
        if self.kept_default:
            if baseline.timing_reuse_key is None:
                raise RuntimeError("default plan has no timing-reuse key")
            self.final = KeptDefaultOutcome(
                selected_plan_sha256=baseline.plan_sha256,
                timing_reuse_key=baseline.timing_reuse_key,
            )
        elif selected is None:
            self.final = NoValidCandidateOutcome()
        elif selected.execution_timed_out:
            self.final = self.timed_out(selected)
        elif selected.timing_reuse_key == baseline.timing_reuse_key:
            if selected.plan_sha256 is None or selected.timing_reuse_key is None:
                raise RuntimeError("validated candidate has no identity")
            self.final = DefaultDuplicateOutcome(
                selected_candidate_id=selected.candidate_id,
                selected_plan_sha256=selected.plan_sha256,
                timing_reuse_key=selected.timing_reuse_key,
            )
        else:
            self.final = self.measure_selected(selected, rng)
        return self.final

    def execute_pair(
        self,
        candidate: Candidate,
        order: tuple[MeasuredPlan, MeasuredPlan],
        *,
        warmup: bool,
    ) -> bool:
        """Retain each completed half; only a candidate timeout is a scored decision."""
        pairs = self.paired.warmups if warmup else self.paired.measurements
        pairs.append(MeasurementPair(order=order))
        for role in order:
            self.check_cancelled()
            self.operation = (
                f"paired_{'warmup' if warmup else 'measurement'}_{role.value}"
            )
            self.execution_counts.final_paired += 1
            if role == MeasuredPlan.DEFAULT:
                result = self.worker.explain(
                    self.sql, self.default_timeout_ms, analyze=True
                )
            else:
                try:
                    result = self.worker.explain(
                        self.sql,
                        self.timeout_ms,
                        analyze=True,
                        hint=candidate.compiled_hint,
                    )
                except QueryTimeoutError as error:
                    self.mark_timeout(candidate, error)
                    self.check_cancelled()
                    return False
            observation = measured(result)
            pairs[-1] = pairs[-1].model_copy(update={role.value: observation})
            self.check_cancelled()
            if role == MeasuredPlan.CANDIDATE:
                self.verify_execution(candidate, result, observation)
            expected = (
                self.default.plan_sha256
                if role == MeasuredPlan.DEFAULT and self.default is not None
                else candidate.plan_sha256
            )
            if observation.plan_sha256 != expected:
                raise PostgresError(
                    f"{role.value} plan changed during paired measurement"
                )
        return True

    def measure_selected(self, candidate: Candidate, rng: random.Random) -> Outcome:
        """Warm both plans and randomize each measured pair; report the raw median ratio."""
        if candidate.plan_sha256 is None or candidate.timing_reuse_key is None:
            raise RuntimeError("validated candidate has no identity")
        for _ in range(self.measurement.paired_warmups):
            if not self.execute_pair(
                candidate, (MeasuredPlan.CANDIDATE, MeasuredPlan.DEFAULT), warmup=True
            ):
                return self.timed_out(candidate)
        for _ in range(self.measurement.paired_measurements):
            order = [MeasuredPlan.CANDIDATE, MeasuredPlan.DEFAULT]
            rng.shuffle(order)
            if not self.execute_pair(candidate, (order[0], order[1]), warmup=False):
                return self.timed_out(candidate)
        default_median = statistics.median(
            pair.default.execution_time_ms
            for pair in self.paired.measurements
            if pair.default is not None
        )
        candidate_median = statistics.median(
            pair.candidate.execution_time_ms
            for pair in self.paired.measurements
            if pair.candidate is not None
        )
        if default_median <= 0 or candidate_median <= 0:
            raise PostgresError("paired median execution times must be positive")
        self.candidates[self.candidates.index(candidate)] = candidate.model_copy(
            update={"measurement_status": MeasurementStatus.MEASURED}
        )
        return MeasuredOutcome(
            selected_candidate_id=candidate.candidate_id,
            selected_plan_sha256=candidate.plan_sha256,
            timing_reuse_key=candidate.timing_reuse_key,
            paired=self.paired,
            default_median_execution_time_ms=default_median,
            candidate_median_execution_time_ms=candidate_median,
            speedup=default_median / candidate_median,
        )

    def mark_timeout(self, candidate: Candidate, error: QueryTimeoutError) -> None:
        """Update the selected attempt without treating the cutoff as a completion time."""
        self.candidates[self.candidates.index(candidate)] = candidate.model_copy(
            update={
                "execution_timed_out": True,
                "timeout_ms": error.timeout_ms,
                "errors_or_diagnostics": [*candidate.errors_or_diagnostics, str(error)],
            }
        )

    def timed_out(self, candidate: Candidate) -> TimedOutOutcome:
        """Retain initial reference and cutoff independently from partial paired readings."""
        if self.default is None or self.default.median_execution_time_ms is None:
            raise RuntimeError("rollout baseline has not been measured")
        return TimedOutOutcome(
            selected_candidate_id=candidate.candidate_id,
            selected_plan_sha256=candidate.plan_sha256,
            timing_reuse_key=candidate.timing_reuse_key,
            initial_default_median_execution_time_ms=self.default.median_execution_time_ms,
            timeout_ms=candidate.timeout_ms
            if candidate.execution_timed_out and candidate.timeout_ms is not None
            else self.timeout_ms,
            paired=self.paired,
        )

    def record(self, error: BaseException | None = None) -> RolloutRecord:
        """Build validated evidence for a completed outcome or an unscored failure."""
        return RolloutRecord(
            task_id=self.task.task_id,
            template_id=self.task.template_id,
            measurement=self.measurement,
            selection=self.selection if self.selection.status != "pending" else None,
            execution_counts=self.execution_counts,
            default=self.default,
            candidates=self.candidates,
            final=self.final if error is None else None,
            failure=RolloutFailure(
                operation=self.operation,
                error_type=type(error).__name__,
                error=str(error),
                paired=self.paired,
            )
            if error is not None
            else None,
        )
