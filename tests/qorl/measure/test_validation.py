"""Plan-only action validation, independent of models, Docker and query timing."""

from dataclasses import dataclass

import pytest
from pydantic import JsonValue

from qorl.measure.schemas import MeasurementStatus
from qorl.measure.validation import PlanValidationEvaluator
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.postgres.exceptions import PostgresError, QueryTimeout
from qorl.postgres.schemas import ExplainResult, PostgresIndexes
from qorl.taskset.schemas import Relation, Task

TIMEOUT_MS = 5_000
ATTEMPT_LIMIT = 3
TASK = Task(
    task_id="job-test",
    template_id="job-test",
    sql_path="test.sql",
    sql_sha256="unused",
    tables=["table_a", "table_b"],
    relations=[
        Relation(alias="a", table="table_a"),
        Relation(alias="b", table="table_b"),
    ],
    join_edges=["a:table_a.id=b:table_b.a_id"],
    table_count=2,
    relation_count=2,
    join_predicate_count=1,
)
SQL = "SELECT * FROM table_a a, table_b b WHERE a.id = b.a_id;"
DEFAULT_PLAN: dict[str, JsonValue] = {
    "Node Type": "Hash Join",
    "Plans": [
        {"Node Type": "Seq Scan", "Alias": "a"},
        {"Node Type": "Seq Scan", "Alias": "b"},
    ],
}
REVERSED_PLAN: dict[str, JsonValue] = {
    "Node Type": "Hash Join",
    "Plans": [
        {"Node Type": "Seq Scan", "Alias": "b"},
        {"Node Type": "Seq Scan", "Alias": "a"},
    ],
}
LEADING: dict[str, JsonValue] = {"version": 1, "leading": {"left": "b", "right": "a"}}
USED_HINT = "HintStateDump: {used hints:Leading((b a))}, {not used hints:(none)}, {duplicate hints:(none)}, {error hints:(none)}"
UNUSED_HINT = "HintStateDump: {used hints:(none)}, {not used hints:Leading((b a))}, {duplicate hints:(none)}, {error hints:(none)}"


class SqlFixture:
    def load_sql(self, task: Task) -> str:
        assert task == TASK
        return SQL


@dataclass(frozen=True)
class ExplainCall:
    sql: str
    timeout_ms: int
    analyze: bool
    hint: str


class Worker:
    def __init__(self, *responses: ExplainResult | PostgresError) -> None:
        self.indexes = PostgresIndexes(by_table={})
        self.responses = iter(responses)
        self.calls: list[ExplainCall] = []

    def explain(
        self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        self.calls.append(ExplainCall(sql, timeout_ms, analyze, hint))
        response = next(self.responses)
        if isinstance(response, PostgresError):
            raise response
        return response


def plain(
    plan: dict[str, JsonValue] = DEFAULT_PLAN, diagnostics: str = ""
) -> ExplainResult:
    return ExplainResult({"Plan": plan}, diagnostics)


def evaluator(
    worker: Worker, attempts: int = ATTEMPT_LIMIT
) -> PlanValidationEvaluator[Worker]:
    return PlanValidationEvaluator(
        worker,
        SqlFixture(),
        TASK,
        default_timeout_ms=TIMEOUT_MS,
        max_candidates=attempts,
    )


def test_leading_action_to_compiled_hint_to_novel_plan_without_execution() -> None:
    worker = Worker(plain(), plain(REVERSED_PLAN, USED_HINT))
    run = evaluator(worker)
    baseline = run.start()
    candidate = run.evaluate(LEADING)

    assert baseline.measurements == [] and baseline.median_execution_time_ms is None
    assert candidate.candidate_id == "candidate-01"
    assert candidate.compiled_hint == "/*+ Leading((b a)) */"
    assert candidate.action_valid and candidate.constraints_satisfied
    assert candidate.structurally_novel
    assert candidate.structural_duplicate_of is None and candidate.duplicate_of is None
    assert candidate.plan_fingerprint_version == PLAN_FINGERPRINT_VERSION
    assert candidate.plain_explain == {"Plan": REVERSED_PLAN}
    assert candidate.pg_hint_plan is not None
    assert candidate.pg_hint_plan["used"] == "Leading((b a))"
    assert candidate.errors_or_diagnostics == []
    assert candidate.measurement_status == MeasurementStatus.NOT_MEASURED
    assert "provisional_speedup" not in candidate.feedback()
    assert worker.calls == [
        ExplainCall(SQL, TIMEOUT_MS, False, ""),
        ExplainCall(SQL, TIMEOUT_MS, False, candidate.compiled_hint),
    ]


def test_malformed_attempt_consumes_id_but_not_sql_or_fake_plan_evidence() -> None:
    worker = Worker(plain(), plain())
    run = evaluator(worker)
    run.start()
    malformed = run.evaluate({"version": 1, "leading": "a"})
    assert not malformed.action_valid and not malformed.constraints_satisfied
    assert malformed.plan_sha256 is None and malformed.timing_reuse_key is None
    assert malformed.errors_or_diagnostics == [
        "leading must contain at least two relations"
    ]
    assert len(worker.calls) == 1
    duplicate = run.evaluate({"version": 1})
    assert duplicate.candidate_id == "candidate-02"
    assert duplicate.attempts_remaining == 1
    assert duplicate.duplicate_of == "default"
    assert duplicate.structural_duplicate_of == "default"
    assert not duplicate.structurally_novel


def test_unused_hint_preserves_postgres_evidence_but_is_not_accepted() -> None:
    worker = Worker(plain(), plain(REVERSED_PLAN, UNUSED_HINT))
    run = evaluator(worker)
    run.start()
    candidate = run.evaluate(LEADING)
    assert candidate.action_valid and not candidate.constraints_satisfied
    assert candidate.plan_sha256 is not None
    assert candidate.plain_explain == {"Plan": REVERSED_PLAN}
    assert candidate.errors_or_diagnostics == [
        "pg_hint_plan reported not used hints: Leading((b a))"
    ]
    assert not candidate.structurally_novel


@pytest.mark.parametrize("changed_estimate", [False, True])
def test_structural_duplicate_does_not_imply_execution_equivalence(
    changed_estimate: bool,
) -> None:
    plan = {**DEFAULT_PLAN, **({"Plan Rows": 9} if changed_estimate else {})}
    worker = Worker(plain(), plain(plan, USED_HINT), plain(plan, USED_HINT))
    run = evaluator(worker)
    baseline = run.start()
    action: dict[str, JsonValue] = {
        "version": 1,
        "settings": {"enable_material": False},
    }
    candidate = run.evaluate(action)
    repeat = run.evaluate(action)
    assert candidate.structural_plan_sha256 == baseline.structural_plan_sha256
    assert candidate.structural_duplicate_of == "default"
    assert candidate.timing_reuse_key != baseline.timing_reuse_key
    assert candidate.duplicate_of is None
    assert repeat.duplicate_of == candidate.candidate_id
    assert repeat.structural_duplicate_of == "default"
    assert (candidate.plan_sha256 != baseline.plan_sha256) == changed_estimate
    assert all(not call.analyze for call in worker.calls)


def test_repeated_novel_plan_has_separate_structural_and_execution_duplicates() -> None:
    worker = Worker(
        plain(), plain(REVERSED_PLAN, USED_HINT), plain(REVERSED_PLAN, USED_HINT)
    )
    run = evaluator(worker)
    run.start()
    first, repeat = run.evaluate(LEADING), run.evaluate(LEADING)
    assert first.structurally_novel
    assert not repeat.structurally_novel
    assert repeat.structural_duplicate_of == first.candidate_id
    assert repeat.duplicate_of == first.candidate_id


def test_candidate_budget_and_default_decision_are_enforced_without_extra_sql() -> None:
    worker = Worker(plain(), plain())
    run = evaluator(worker, attempts=1)
    with pytest.raises(RuntimeError, match="not been started"):
        run.evaluate({"version": 1})
    run.start()
    with pytest.raises(RuntimeError, match="already been started"):
        run.start()
    run.evaluate({"version": 1})
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        run.evaluate({"version": 1})
    with pytest.raises(RuntimeError, match="before submitting"):
        run.keep_default()
    assert len(worker.calls) == 2


def test_keep_default_closes_candidate_submission() -> None:
    worker = Worker(plain())
    run = evaluator(worker)
    run.start()
    assert run.keep_default() == {"status": "kept_default"}
    with pytest.raises(RuntimeError, match="already kept"):
        run.evaluate({"version": 1})
    with pytest.raises(RuntimeError, match="already kept"):
        run.keep_default()
    assert len(worker.calls) == 1


def test_planning_timeout_is_not_a_completed_latency_measurement() -> None:
    worker = Worker(plain(), QueryTimeout(TIMEOUT_MS))
    run = evaluator(worker)
    run.start()
    candidate = run.evaluate(LEADING)
    assert candidate.execution_timed_out
    assert candidate.timeout_ms == TIMEOUT_MS
    assert candidate.action_valid and not candidate.constraints_satisfied
    assert candidate.plan_sha256 is None and candidate.plain_explain is None
    assert "provisional_speedup" not in candidate.feedback()
    assert candidate.measurement_status == MeasurementStatus.NOT_MEASURED


@pytest.mark.parametrize("during_default", [False, True])
def test_database_failure_propagates_instead_of_becoming_an_invalid_action(
    during_default: bool,
) -> None:
    error = PostgresError("connection lost")
    worker = Worker(error) if during_default else Worker(plain(), error)
    run = evaluator(worker)
    with pytest.raises(PostgresError, match="connection lost"):
        run.start()
        run.evaluate(LEADING)
    assert run.candidates == []
