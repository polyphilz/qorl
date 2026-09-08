"""Intermediate evidence is observable work, distinct from final paired scoring."""

import json
import random
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.qorl.agent.test_agent import ScriptedTransport, policy, reply
from tests.qorl.measure.test_rollout import (
    ACTION,
    SETTINGS,
    TASK,
    Sql,
    Worker,
    evaluator,
)

from qorl.agent.feedback import execution_observation
from qorl.agent.interface import AgentInterface
from qorl.agent.presentation import FIELD_BYTES, PLAN_BYTES, plan_view
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor
from qorl.evaluation.evaluate import summarize_performance
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import RolloutMeasurementSettings, RolloutRecord
from qorl.measure.validation import PlanValidationEvaluator
from qorl.model.client import JSON_OBJECT
from qorl.model.schemas import JsonObject, JsonValue
from qorl.plans.fingerprint import plan_sha256
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import ExplainResult, WorkerAllocation

ENABLED = SETTINGS.model_copy(
    update={"candidate_feedback_warmups": 1, "candidate_feedback_measurements": 1}
)

EXECUTION_DIAGNOSTICS: JsonObject = {
    "Full-sort Groups": {
        "Group Count": 3,
        "Sort Methods Used": ["quicksort", "external merge"],
        "Sort Space Memory": {
            "Average Sort Space Used": 25,
            "Peak Sort Space Used": 30,
        },
        "Sort Space Disk": {"Average Sort Space Used": 50, "Peak Sort Space Used": 60},
    },
    "Pre-sorted Groups": {
        "Group Count": 2,
        "Sort Methods Used": ["external merge"],
        "Sort Space Disk": {"Average Sort Space Used": 70, "Peak Sort Space Used": 80},
    },
    "Cache Hits": 23,
    "Cache Misses": 7,
    "Cache Evictions": 2,
    "Cache Overflows": 1,
    "HashAgg Batches": 4,
}


@pytest.mark.parametrize("summary", [True, False])
@pytest.mark.parametrize(("field", "value"), EXECUTION_DIAGNOSTICS.items())
def test_observed_diagnostics_preserve_values_only_when_present(
    summary: bool, field: str, value: JsonValue
) -> None:
    plan: JsonObject = {"Node Type": "Result", field: value}
    observed = plan_view(plan, summary=summary, observed=True)
    assert observed.nodes[0].observed == {field: value}
    assert observed.nodes[0].omitted_fields == []
    missing = plan_view({"Node Type": "Result"}, summary=summary, observed=True)
    assert missing.nodes[0].observed == {} and missing.nodes[0].omitted_fields == []
    estimates = plan_view(plan, summary=summary)
    assert estimates.nodes[0].observed is None
    assert field not in estimates.model_dump_json()


@pytest.mark.parametrize("summary", [True, False])
@pytest.mark.parametrize("field", ["Full-sort Groups", "Pre-sorted Groups"])
def test_oversized_nested_diagnostics_are_explicitly_omitted(
    summary: bool, field: str
) -> None:
    plan: JsonObject = {
        "Node Type": "Incremental Sort",
        field: {"Sort Space Disk": {"oversized": "x" * FIELD_BYTES}},
    }
    view = plan_view(plan, summary=summary, observed=True)
    assert view.nodes[0].observed == {}
    assert view.nodes[0].omitted_fields == [field]
    assert len(json.dumps(view.model_dump(mode="json")).encode()) <= PLAN_BYTES


@pytest.mark.parametrize("summary", [True, False])
def test_nested_diagnostics_respect_whole_plan_bound(summary: bool) -> None:
    group: JsonObject = {"Sort Methods Used": ["x" * 1500]}
    assert len(json.dumps(group).encode()) < FIELD_BYTES
    node: JsonObject = {
        "Node Type": "Incremental Sort",
        "Full-sort Groups": group,
        "Pre-sorted Groups": group,
    }
    children: list[JsonValue] = [node for _ in range(40)]
    view = plan_view(node | {"Plans": children}, summary=summary, observed=True)
    assert 0 < len(view.nodes) < 16  # Byte limit, before either node-count limit.
    assert view.omitted_nodes == 41 - len(view.nodes)
    assert len(json.dumps(view.model_dump(mode="json")).encode()) <= PLAN_BYTES
    assert all(item.omitted_fields == [] for item in view.nodes)
    assert all(
        item.observed == {"Full-sort Groups": group, "Pre-sorted Groups": group}
        for item in view.nodes
    )


@pytest.mark.parametrize(
    ("warmups", "measurements", "valid"),
    [
        (0, 0, True),
        (0, 1, True),
        (1, 1, True),
        (2, 3, True),
        (1, 0, False),
        (-1, 1, False),
        (0, -1, False),
    ],
)
def test_feedback_count_contract(warmups: int, measurements: int, valid: bool) -> None:
    data = SETTINGS.model_dump() | {
        "candidate_feedback_warmups": warmups,
        "candidate_feedback_measurements": measurements,
    }
    if valid:
        result = RolloutMeasurementSettings.model_validate(data)
        assert result.candidate_feedback_measurements == measurements
    else:
        with pytest.raises(ValidationError):
            RolloutMeasurementSettings.model_validate(data)


@pytest.mark.parametrize(
    ("settings", "attempts", "expected"),
    [(SETTINGS, 1, 10), (ENABLED, 1, 12), (ENABLED, 5, 20)],
)
def test_feedback_work_and_fresh_selected_pairs(
    settings: RolloutMeasurementSettings, attempts: int, expected: int
) -> None:
    worker = Worker()
    run = evaluator(worker, settings=settings, attempts=attempts)
    run.start()
    for index in range(attempts):
        worker.candidate_ms = float(index + 1)
        candidate = run.evaluate(
            {"version": 1, "settings": {"seq_page_cost": index + 2.0}}
        )
        if settings.candidate_feedback_measurements:
            assert candidate.execution_feedback is not None
            assert (
                candidate.execution_feedback.measurements[0].execution_time_ms
                == index + 1
            )
    worker.candidate_ms = (
        25.0  # Final measurements are fresh, not the preliminary observations.
    )
    final = run.finish(random.Random(1), selected_candidate_id="candidate-01")
    assert final.kind == "measured" and final.speedup == 0.4
    assert sum(call.analyze for call in worker.calls) == expected
    assert run.execution_counts.model_dump() == {
        "initial_default": 2,
        "candidate_feedback": 2 * attempts
        if settings.candidate_feedback_measurements
        else 0,
        "final_paired": 8,
    }
    report = summarize_performance([run.record()])
    assert (
        report.execution_counts == run.execution_counts
        and report.execution_accounting_missing == 0
    )
    assert (
        RolloutRecord.model_validate_json(run.record().model_dump_json())
        == run.record()
    )


class InspectionWorker(Worker):
    def __init__(
        self, repository: Path, *, cancel: Event | None = None, drift: str = ""
    ) -> None:
        super().__init__()
        self.settings = PostgresConfig.load(
            repository / "docker/postgres/configs/000-pgconf-default"
        ).agent_settings
        self.allocation: WorkerAllocation | None = None
        self.cancel = cancel
        self.drift = drift

    def admin_sql(self, sql: str) -> str:
        raise AssertionError("no inspection SQL expected")

    def explain(
        self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        result = super().explain(sql, timeout_ms, analyze=analyze, hint=hint)
        if analyze:
            result.document["Plan"].update(EXECUTION_DIAGNOSTICS)
            result.document["Plan"].update(
                {
                    "Actual Rows": 42,
                    "Actual Loops": 2,
                    "Temp Written Blocks": 7,
                    "Sort Method": "external merge",
                    "Sort Space Type": "Disk",
                    "Sort Space Used": 32,
                    "Workers Launched": 2,
                }
            )
            if hint:
                if self.cancel is not None:
                    self.cancel.set()
                if self.drift == "plan":
                    result.document["Plan"]["Plan Rows"] = 987
                if self.drift == "hints":
                    return ExplainResult(
                        result.document,
                        "HintStateDump: {used hints:(none)}, {not used hints:Set(seq_page_cost)}, {duplicate hints:(none)}, {error hints:(none)}",
                    )
        return result


@pytest.mark.parametrize("attempts", [1, 5])
@pytest.mark.parametrize("measurement", [None, SETTINGS, ENABLED])
def test_tool_description_matches_active_evaluator(
    repository_root: Path,
    attempts: int,
    measurement: RolloutMeasurementSettings | None,
) -> None:
    worker = InspectionWorker(repository_root)
    run: PlanValidationEvaluator[InspectionExecutor]
    if measurement is None:
        run = PlanValidationEvaluator(
            worker, Sql(), TASK, default_timeout_ms=5000, max_candidates=attempts
        )
    else:
        run = RolloutEvaluator(
            worker, Sql(), TASK, measurement=measurement, max_candidates=attempts
        )
    run.start()
    interface = AgentInterface.from_evaluator(run, 64)
    expected = "Submit one self-contained PlanAction for plain-EXPLAIN validation and return validation diagnostics."
    if measurement is not None and measurement.candidate_feedback_measurements > 0:
        expected += " For valid candidates, also return available execution observations, including timings and observed plan diagnostics."
    reference = agent_tools(["a", "b"], execution_feedback=False)
    assert len(interface.tools) == len(reference) == 6
    for tool, original in zip(interface.tools, reference, strict=True):
        assert tool.function.name == original.function.name
        assert tool.function.parameters == original.function.parameters
        if tool.function.name == "evaluate_candidate":
            assert tool.function.description == expected
        else:
            assert tool == original


def test_feedback_is_in_next_request_even_when_only_finish_remains(
    repository_root: Path,
) -> None:
    worker = InspectionWorker(repository_root)
    run = RolloutEvaluator[InspectionExecutor](
        worker, Sql(), TASK, measurement=ENABLED, max_candidates=1
    )
    run.start()
    transport = ScriptedTransport(
        [reply("evaluate_candidate", json.dumps({"action": ACTION})), reply("finish")]
    )
    trace = policy(transport).search(run)
    assert run.final is None and run.execution_counts.candidate_feedback == 2
    messages = TypeAdapter(list[JsonObject]).validate_python(
        transport.requests[1]["messages"]
    )
    tool_message = next(message for message in messages if message["role"] == "tool")
    assert isinstance(tool_message["content"], str)
    feedback = JSON_OBJECT.validate_python(
        JSON_OBJECT.validate_json(tool_message["content"])["execution_feedback"]
    )
    assert feedback["median_execution_time_ms"] == 5.0
    assert "Actual Rows" not in trace.initial_observation.model_dump_json()
    environment = AgentEnvironment(run)
    default, _ = environment.execute("get_plan", {"candidate_id": "default"})
    assert "Actual Rows" not in json.dumps(default)
    before = len(worker.calls)
    detail, _ = environment.execute("get_plan", {"candidate_id": "candidate-01"})
    assert len(worker.calls) == before
    assert '"Actual Rows": 42' in json.dumps(detail)
    assert '"Temp Written Blocks": 7' in json.dumps(detail)
    assert '"Workers Launched": 2' in json.dumps(detail)
    assert detail["displayed_sample_phase"] == "measurement"
    assert detail["displayed_sample_index"] == 0
    # The initial trace and default get_plan remain estimate-only even though
    # retained baseline samples contain these same execution diagnostics.
    view = execution_observation(
        run.candidates[0], run.default, run.candidates, summary=False
    )
    assert view is not None and view.plan is not None
    assert view.plan.nodes[0].observed is not None
    for field, value in EXECUTION_DIAGNOSTICS.items():
        assert field not in trace.initial_observation.model_dump_json()
        assert field not in json.dumps(default)
        assert view.plan.nodes[0].observed[field] == value
    assert detail == JSON_OBJECT.validate_python(view.model_dump(mode="json"))


def test_feedback_warmups_excluded_and_reuse_is_exact() -> None:
    class VariableTimingWorker(Worker):
        def explain(
            self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
        ) -> ExplainResult:
            self.candidate_ms = (
                5.0
                if any(call.analyze and call.candidate for call in self.calls)
                else 1000.0
            )
            return super().explain(sql, timeout_ms, analyze=analyze, hint=hint)

    worker = VariableTimingWorker()
    run = evaluator(worker, settings=ENABLED, attempts=4)
    baseline = run.start()
    first = run.evaluate(ACTION)
    assert first.execution_feedback is not None
    assert first.execution_feedback.warmups[0].execution_time_ms == 1000.0
    view = execution_observation(first, baseline, run.candidates)
    assert view is not None and view.median_execution_time_ms == 5.0
    assert view.preliminary_ratio_to_initial_default == 2.0
    repeated = run.evaluate(ACTION)
    assert (
        repeated.execution_feedback is not None
        and repeated.execution_feedback.source_id == first.candidate_id
    )
    assert (
        repeated.execution_feedback.warmups
        == repeated.execution_feedback.measurements
        == []
    )
    reused = execution_observation(repeated, baseline, run.candidates)
    assert (
        reused is not None
        and reused.new_executions == 0
        and reused.measurement_count == 1
    )
    default = run.evaluate({"version": 1})
    assert (
        default.execution_feedback is not None
        and default.execution_feedback.source_id == "default"
    )
    baseline_view = execution_observation(default, baseline, run.candidates)
    assert (
        baseline_view is not None
        and baseline_view.reused
        and baseline_view.measurement_count == 1
    )
    changed = run.evaluate({"version": 1, "settings": {"seq_page_cost": 3.0}})
    assert changed.structural_plan_sha256 == first.structural_plan_sha256
    assert changed.timing_reuse_key != first.timing_reuse_key
    assert (
        changed.execution_feedback is not None
        and changed.execution_feedback.source_id is None
    )
    assert run.execution_counts.candidate_feedback == 4


@pytest.mark.parametrize("fail_execution", [1, 2, 3])
def test_timeout_retains_cutoff_partial_evidence_and_reuses_without_retry(
    fail_execution: int,
) -> None:
    worker = Worker(failure="candidate_execution", fail_execution=fail_execution)
    settings = ENABLED.model_copy(update={"candidate_feedback_measurements": 2})
    run = evaluator(worker, settings=settings, attempts=3)
    run.start()
    first = run.evaluate(ACTION)
    assert (
        first.execution_feedback is not None
        and first.execution_feedback.status == "timed_out"
    )
    assert len(first.execution_feedback.warmups) == min(1, fail_execution - 1)
    assert len(first.execution_feedback.measurements) == max(0, fail_execution - 2)
    view = execution_observation(first, run.default, run.candidates)
    assert view is not None and view.timeout_ms == run.timeout_ms
    assert (
        view.median_execution_time_ms is None
        and view.preliminary_ratio_to_initial_default is None
    )
    repeated = run.evaluate(ACTION)
    assert (
        repeated.execution_timed_out
        and run.execution_counts.candidate_feedback == fail_execution
    )
    repaired = run.evaluate({"version": 1, "settings": {"seq_page_cost": 3.0}})
    assert not repaired.execution_timed_out
    result = run.finish(random.Random(0), selected_candidate_id=first.candidate_id)
    assert result.kind == "timed_out" and run.paired.measurements == []
    run.record()


@pytest.mark.parametrize("drift", ["plan", "hints"])
def test_feedback_drift_is_unscored_and_completed_document_survives(
    repository_root: Path, drift: str
) -> None:
    run = evaluator(InspectionWorker(repository_root, drift=drift), settings=ENABLED)
    run.start()
    with pytest.raises(PostgresError) as caught:
        run.evaluate(ACTION)
    record = run.record(caught.value)
    feedback = record.candidates[0].execution_feedback
    assert feedback is not None and len(feedback.warmups) == 1
    assert feedback.warmups[0].analyzed_document is not None
    assert feedback.warmups[0].hint_diagnostics is not None
    assert record.final is None and record.failure is not None
    assert run.execution_counts.candidate_feedback == 1


def test_feedback_cancellation_retains_completed_analysis(
    repository_root: Path,
) -> None:
    cancel = Event()
    worker = InspectionWorker(repository_root, cancel=cancel)
    run = RolloutEvaluator(
        worker, Sql(), TASK, measurement=ENABLED, max_candidates=1, cancel=cancel
    )
    run.start()
    with pytest.raises(CancelledError) as caught:
        run.evaluate(ACTION)
    record = run.record(caught.value)
    feedback = record.candidates[0].execution_feedback
    assert feedback is not None and len(feedback.warmups) == 1
    document = feedback.warmups[0].analyzed_document
    assert (
        document is not None
        and plan_sha256(JSON_OBJECT.validate_python(document["Plan"]))
        == record.candidates[0].plan_sha256
    )
    assert (
        record.failure is not None
        and record.failure.operation == "candidate_feedback_warmup"
    )


@pytest.mark.parametrize(
    "action", [{}, {"version": 1, "scans": [{"relation": "a", "force": "index"}]}]
)
def test_rejected_attempt_does_not_execute_feedback(action: JsonObject) -> None:
    worker = Worker()
    run = evaluator(worker, settings=ENABLED)
    run.start()
    candidate = run.evaluate(action)
    assert not candidate.constraints_satisfied and candidate.execution_feedback is None
    assert (
        run.execution_counts.candidate_feedback == 0
        and sum(call.analyze for call in worker.calls) == 2
    )


def test_baseline_feedback_reuse_preserves_final_duplicate_outcome() -> None:
    worker = Worker()
    run = evaluator(worker, settings=ENABLED)
    run.start()
    candidate = run.evaluate({"version": 1})
    view = execution_observation(candidate, run.default, run.candidates)
    assert view is not None and view.source_id == "default"
    assert view.new_executions == 0 and view.warmup_count == view.measurement_count == 1
    assert run.finish(random.Random(0)).kind == "default_duplicate"
    assert sum(call.analyze for call in worker.calls) == 2
    assert (
        run.execution_counts.candidate_feedback
        == run.execution_counts.final_paired
        == 0
    )
