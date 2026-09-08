import json
import random
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest
from pydantic import JsonValue, ValidationError

from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import (
    DefaultDuplicateOutcome,
    KeptDefaultOutcome,
    MeasuredOutcome,
    NoValidCandidateOutcome,
    RolloutMeasurementSettings,
    RolloutRecord,
    TimedOutOutcome,
)
from qorl.postgres.exceptions import PostgresError, QueryTimeoutError
from qorl.postgres.schemas import ExplainResult, PostgresIndexes
from qorl.taskset.schemas import Task

SETTINGS = RolloutMeasurementSettings(
    default_warmups=1,
    default_measurements=1,
    paired_warmups=1,
    paired_measurements=3,
    default_timeout_seconds=300.0,
    candidate_timeout_floor_seconds=5.0,
    candidate_timeout_multiplier=3.0,
)
TASK = Task.model_validate(
    {
        "task_id": "job-test",
        "template_id": "job-test-template",
        "sql_path": "queries/test.sql",
        "sql_sha256": "unused",
        "tables": ["table_a", "table_b"],
        "relations": [
            {"alias": "a", "table": "table_a"},
            {"alias": "b", "table": "table_b"},
        ],
        "join_edges": ["a:table_a.id=b:table_b.a_id"],
        "table_count": 2,
        "relation_count": 2,
        "join_predicate_count": 1,
    }
)
PLAN: dict[str, JsonValue] = {
    "Node Type": "Hash Join",
    "Plans": [
        {"Node Type": "Seq Scan", "Alias": "a"},
        {"Node Type": "Seq Scan", "Alias": "b"},
    ],
}
ACTION: dict[str, JsonValue] = {"version": 1, "settings": {"seq_page_cost": 2.0}}


class Sql:
    def load_sql(self, task: Task) -> str:
        assert task == TASK
        return "SELECT 1;"


@dataclass(frozen=True)
class Call:
    analyze: bool
    candidate: bool
    timeout_ms: int


class Worker:
    def __init__(
        self,
        *,
        default_ms: float = 10.0,
        candidate_ms: float = 5.0,
        failure: str = "",
        fail_execution: int = 1,
        changed_structure: bool = False,
        changed_estimates: bool = False,
    ) -> None:
        self.indexes = PostgresIndexes(
            by_table={"table_a": frozenset(), "table_b": frozenset()}
        )
        self.calls: list[Call] = []
        self.default_ms = default_ms
        self.candidate_ms = candidate_ms
        self.failure = failure
        self.fail_execution = fail_execution
        self.changed_structure = changed_structure
        self.changed_estimates = changed_estimates

    def explain(
        self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        assert sql == "SELECT 1;"
        self.calls.append(Call(analyze, bool(hint), timeout_ms))
        role = "candidate" if hint else "default"
        phase = "execution" if analyze else "plan"
        count = sum(
            call.analyze == analyze and call.candidate == bool(hint)
            for call in self.calls
        )
        if self.failure == f"{role}_{phase}" and count == self.fail_execution:
            raise QueryTimeoutError(timeout_ms)
        if self.failure == "infrastructure" and hint:
            raise PostgresError("connection lost")
        plan = deepcopy(PLAN)
        if hint and self.changed_structure:
            plan["Node Type"] = "Merge Join"
        if hint and self.changed_estimates:
            plan["Plan Rows"] = 999
        document: dict[str, JsonValue] = {"Plan": plan}
        if analyze:
            document.update(
                {
                    "Execution Time": self.candidate_ms if hint else self.default_ms,
                    "Planning Time": 1.0,
                }
            )
        diagnostics = (
            "HintStateDump: {used hints:Set(seq_page_cost)}, {not used hints:(none)}, {duplicate hints:(none)}, {error hints:(none)}"
            if hint
            else ""
        )
        return ExplainResult(document, diagnostics)


def evaluator(
    worker: Worker,
    *,
    attempts: int = 1,
    settings: RolloutMeasurementSettings = SETTINGS,
) -> RolloutEvaluator[Worker]:
    return RolloutEvaluator(
        worker, Sql(), TASK, measurement=settings, max_candidates=attempts
    )


@pytest.mark.parametrize(
    "structure,estimates", [(False, False), (False, True), (True, False)]
)
def test_valid_candidate_needs_timing_when_reuse_key_differs(
    structure: bool, estimates: bool
) -> None:
    worker = Worker(changed_structure=structure, changed_estimates=estimates)
    run = evaluator(worker)
    baseline = run.start()
    candidate = run.evaluate(ACTION)
    assert candidate.constraints_satisfied
    assert candidate.structurally_novel == structure
    assert candidate.timing_reuse_key != baseline.timing_reuse_key
    assert sum(call.analyze for call in worker.calls) == 2
    final = run.finish(random.Random(0))
    assert isinstance(final, MeasuredOutcome)
    assert final.speedup == 2.0
    assert len(final.paired.warmups) == 1
    assert len(final.paired.measurements) == 3
    assert sum(call.analyze for call in worker.calls) == 10
    assert sum(not call.analyze for call in worker.calls) == 2
    assert (
        RolloutRecord.model_validate_json(run.record().model_dump_json())
        == run.record()
    )


@pytest.mark.parametrize("decision", ["keep_default", "duplicate", "invalid"])
def test_branches_without_candidate_measurements(decision: str) -> None:
    worker = Worker()
    run = evaluator(worker)
    baseline = run.start()
    if decision == "keep_default":
        run.keep_default()
    else:
        candidate = run.evaluate({"version": 1 if decision == "duplicate" else 2})
        assert candidate.attempts_remaining == 0
    final = run.finish(random.Random(0))
    assert sum(call.analyze for call in worker.calls) == 2
    if decision == "keep_default":
        assert isinstance(final, KeptDefaultOutcome)
        assert final.selected_candidate_id is None
    elif decision == "duplicate":
        assert isinstance(final, DefaultDuplicateOutcome)
        assert final.selected_plan_sha256 == baseline.plan_sha256
    else:
        assert isinstance(final, NoValidCandidateOutcome)
        assert final.speedup is None
    assert "paired" not in final.to_wire()
    run.record()


@pytest.mark.parametrize(
    "failure,execution",
    [
        ("candidate_plan", 1),
        ("candidate_execution", 1),
        ("candidate_execution", 2),
        ("candidate_execution", 3),
    ],
)
def test_candidate_timeouts_preserve_cutoff_and_partial_evidence(
    failure: str, execution: int
) -> None:
    worker = Worker(failure=failure, fail_execution=execution, default_ms=120_000.0)
    run = evaluator(worker)
    run.start()
    run.evaluate(ACTION)
    final = run.finish(random.Random(0))
    assert isinstance(final, TimedOutOutcome)
    assert final.speedup is None
    assert final.timeout_ms == 360_000
    assert final.initial_default_median_execution_time_ms == 120_000.0
    assert (final.selected_plan_sha256 is None) == (failure == "candidate_plan")
    assert (final.timing_reuse_key is None) == (failure == "candidate_plan")
    candidate_samples = [
        pair.candidate
        for pair in [*final.paired.warmups, *final.paired.measurements]
        if pair.candidate is not None
    ]
    assert len(candidate_samples) == (
        0 if failure == "candidate_plan" else execution - 1
    )
    assert all(sample.execution_time_ms == 5.0 for sample in candidate_samples)
    assert all(
        call.timeout_ms == (360_000 if call.candidate else 300_000)
        for call in worker.calls
    )
    run.record()


@pytest.mark.parametrize(
    "failure,execution",
    [
        ("default_plan", 1),
        ("default_execution", 1),
        ("default_execution", 2),
        ("default_execution", 3),
        ("default_execution", 4),
        ("infrastructure", 1),
    ],
)
def test_default_and_infrastructure_failures_are_unscored(
    failure: str, execution: int
) -> None:
    worker = Worker(failure=failure, fail_execution=execution)
    run = evaluator(worker)
    with pytest.raises(PostgresError) as caught:
        run.start()
        run.evaluate(ACTION)
        run.finish(random.Random(0))
    record = run.record(caught.value)
    assert record.final is None and record.failure is not None
    assert record.failure.error_type in {"QueryTimeoutError", "PostgresError"}
    if failure == "default_execution" and execution == 2:
        assert record.default is not None and len(record.default.warmups) == 1
    if failure == "default_execution" and execution >= 3:
        assert record.failure.paired.warmups[0].candidate is not None
    assert RolloutRecord.model_validate_json(record.model_dump_json()) == record


def test_raw_speedup_is_not_training_clipped() -> None:
    run = evaluator(Worker(default_ms=100.0, candidate_ms=5.0))
    run.start()
    run.evaluate(ACTION)
    assert run.finish(random.Random(0)).speedup == 20.0


def test_counts_are_driven_by_settings() -> None:
    worker = Worker()
    settings = SETTINGS.model_copy(
        update={
            "default_warmups": 2,
            "default_measurements": 3,
            "paired_warmups": 2,
            "paired_measurements": 4,
        }
    )
    run = evaluator(worker, settings=settings)
    run.start()
    run.evaluate(ACTION)
    run.finish(random.Random(0))
    assert sum(call.analyze for call in worker.calls) == 17
    run.record()


def test_multiple_attempts_select_one_result_without_intermediate_timing() -> None:
    worker = Worker(failure="candidate_plan")
    run = evaluator(worker, attempts=4)
    run.start()
    run.evaluate({"version": 2})
    run.evaluate(ACTION)  # Planning timeout, retained as an earlier attempt.
    third = run.evaluate(ACTION)
    fourth = run.evaluate(ACTION)
    assert fourth.duplicate_of == third.candidate_id
    assert sum(call.analyze for call in worker.calls) == 2
    with pytest.raises(
        ValueError, match="selected_candidate_id: required when multiple"
    ):
        run.finish(random.Random(0))
    final = run.finish(random.Random(0), selected_candidate_id=fourth.candidate_id)
    assert isinstance(final, MeasuredOutcome)
    assert final.selected_candidate_id == fourth.candidate_id
    assert run.candidates[1].execution_timed_out
    assert sum(call.analyze for call in worker.calls) == 10
    run.record()


def test_outcomes_require_consistent_evidence() -> None:
    run = evaluator(Worker())
    run.start()
    run.evaluate(ACTION)
    final = run.finish(random.Random(0))
    assert isinstance(final, MeasuredOutcome)
    with pytest.raises(ValidationError, match="unclipped"):
        MeasuredOutcome.model_validate({**final.model_dump(), "speedup": 10.0})
    record = run.record().model_dump()
    record["final"]["selected_candidate_id"] = "invented"
    with pytest.raises(ValidationError, match="recorded attempt"):
        RolloutRecord.model_validate(record)
    with pytest.raises(RuntimeError, match="already started"):
        run.finish(random.Random(0))


def test_reviewed_rollout_record() -> None:
    run = evaluator(Worker())
    run.start()
    run.evaluate(ACTION)
    run.finish(random.Random(0))
    expected = json.loads(Path(__file__).with_name("golden_rollout.json").read_text())
    assert run.record().to_wire() == expected


def test_cancellation_between_queries_remains_unscored() -> None:
    cancel = Event()
    worker = Worker()
    run = RolloutEvaluator(
        worker, Sql(), TASK, measurement=SETTINGS, max_candidates=1, cancel=cancel
    )
    run.start()
    cancel.set()
    with pytest.raises(CancelledError) as caught:
        run.evaluate(ACTION)
    assert len(worker.calls) == 3
    record = run.record(caught.value)
    assert record.final is None and record.failure is not None
    assert record.failure.error_type == "CancelledError"


def test_cancellation_retains_completed_default_explain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = Event()
    worker = Worker()
    original_explain = worker.explain

    def explain(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        result = original_explain(sql, timeout_ms, analyze=analyze, hint=hint)
        cancel.set()
        return result

    monkeypatch.setattr(worker, "explain", explain)
    run = RolloutEvaluator(
        worker, Sql(), TASK, measurement=SETTINGS, max_candidates=1, cancel=cancel
    )
    with pytest.raises(CancelledError) as caught:
        run.start()
    record = run.record(caught.value)
    assert record.final is None and record.failure is not None
    assert record.default is not None and record.default.plain_explain == {"Plan": PLAN}
    assert record.default.measurements == []
    assert len(worker.calls) == 1


@pytest.mark.parametrize("planning_timeout", [False, True])
def test_cancellation_retains_completed_candidate_validation(
    monkeypatch: pytest.MonkeyPatch, planning_timeout: bool
) -> None:
    cancel = Event()
    worker = Worker(failure="candidate_plan" if planning_timeout else "")
    original_explain = worker.explain

    def explain(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        try:
            return original_explain(sql, timeout_ms, analyze=analyze, hint=hint)
        finally:
            if hint:
                cancel.set()

    monkeypatch.setattr(worker, "explain", explain)
    run = RolloutEvaluator(
        worker, Sql(), TASK, measurement=SETTINGS, max_candidates=1, cancel=cancel
    )
    run.start()
    with pytest.raises(CancelledError) as caught:
        run.evaluate(ACTION)
    record = run.record(caught.value)
    assert record.final is None and record.failure is not None
    assert record.failure.error_type == "CancelledError"
    assert len(record.candidates) == 1
    candidate = record.candidates[0]
    assert candidate.action == ACTION
    assert candidate.attempts_remaining == 0
    assert candidate.execution_timed_out == planning_timeout
    if planning_timeout:
        assert candidate.plain_explain is None
        assert candidate.errors_or_diagnostics
    else:
        assert candidate.constraints_satisfied
        assert candidate.plain_explain == {"Plan": PLAN}
        assert candidate.structural_duplicate_of == "default"
        assert candidate.timing_reuse_key is not None
        assert candidate.pg_hint_plan is not None
    assert sum(call.analyze for call in worker.calls) == 2


def test_warmups_do_not_enter_the_paired_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = Worker()
    original_explain = worker.explain
    defaults = iter([900.0, 100.0, 800.0, 90.0, 110.0, 100.0])
    candidates = iter([700.0, 60.0, 40.0, 50.0])

    def explain(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        result = original_explain(sql, timeout_ms, analyze=analyze, hint=hint)
        if analyze:
            result.document["Execution Time"] = next(candidates if hint else defaults)
        return result

    monkeypatch.setattr(worker, "explain", explain)
    run = evaluator(worker)
    assert run.start().median_execution_time_ms == 100.0
    run.evaluate(ACTION)
    final = run.finish(random.Random(0))
    assert isinstance(final, MeasuredOutcome)
    assert final.default_median_execution_time_ms == 100.0
    assert final.candidate_median_execution_time_ms == 50.0
    assert final.speedup == 2.0
    assert final.paired.warmups[0].candidate is not None
    assert final.paired.warmups[0].candidate.execution_time_ms == 700.0
    run.record()


def test_changed_plan_during_measurement_is_an_unscored_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = Worker()
    original_explain = worker.explain

    def explain(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        result = original_explain(sql, timeout_ms, analyze=analyze, hint=hint)
        if analyze and hint:
            result.document["Plan"]["Plan Rows"] = 123.0
        return result

    monkeypatch.setattr(worker, "explain", explain)
    run = evaluator(worker)
    run.start()
    run.evaluate(ACTION)
    with pytest.raises(PostgresError, match="plan changed") as caught:
        run.finish(random.Random(0))
    record = run.record(caught.value)
    assert record.final is None and record.failure is not None
    assert record.failure.paired.warmups[0].candidate is not None
