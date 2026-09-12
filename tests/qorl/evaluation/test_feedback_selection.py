"""Keep proposal evidence fixed while evaluating two final selection policies."""

import json
import random
import subprocess
from pathlib import Path

import pytest
from tests.qorl.evaluation.conftest import EvaluationActivity
from tests.qorl.evaluation.test_evaluate import run_evaluation, saved_rollouts
from tests.qorl.measure.test_rollout import ACTION, SETTINGS, Worker, evaluator

from qorl.evaluation.schemas import EvaluationReport
from qorl.evaluation.selection import (
    best_feedback_candidate,
    feedback_selection_evaluator,
)
from qorl.experiment.schemas import EvaluationExperimentConfig
from qorl.measure.schemas import (
    MeasuredOutcome,
    OutcomeKind,
    RunStatus,
    TimedOutOutcome,
)
from qorl.model.client import HttpTransport
from qorl.model.schemas import JsonObject
from qorl.postgres.config import PostgresConfig
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig


@pytest.mark.parametrize(
    "times,expected",
    [
        ([20.0, 10.0], "default"),
        ([10.0 / 1.049], "default"),
        ([10.0 / 1.05], "candidate-01"),
        ([5.0, 2.0, 2.0], "candidate-02"),
        ([2.0, 2.0, 5.0], "candidate-01"),
    ],
)
def test_rule_threshold_best_probe_and_earliest_tie(
    times: list[float], expected: str
) -> None:
    worker = Worker()
    run = evaluator(
        worker,
        attempts=len(times),
        settings=SETTINGS.model_copy(update={"candidate_feedback_measurements": 1}),
    )
    run.start()
    for i, ms in enumerate(times):
        worker.candidate_ms = ms
        run.evaluate({"version": 1, "settings": {"seq_page_cost": 2.0 + i}})
    assert best_feedback_candidate(run) == expected


@pytest.mark.parametrize("unusable", ["invalid", "unsatisfied", "timeout", "missing"])
def test_rule_excludes_unusable_attempts(unusable: str) -> None:
    run = evaluator(
        Worker(),
        settings=SETTINGS.model_copy(update={"candidate_feedback_measurements": 1}),
    )
    run.start()
    candidate = run.evaluate(ACTION)
    changes = {
        "invalid": {"action_valid": False},
        "unsatisfied": {"constraints_satisfied": False},
        "timeout": {"execution_timed_out": True},
        "missing": {"execution_feedback": None},
    }
    run.candidates[0] = candidate.model_copy(update=changes[unusable])
    assert best_feedback_candidate(run) == "default"


def test_reused_feedback_uses_its_source_without_timing_again() -> None:
    worker = Worker()
    run = evaluator(
        worker,
        attempts=2,
        settings=SETTINGS.model_copy(update={"candidate_feedback_measurements": 1}),
    )
    run.start()
    first = run.evaluate(ACTION)
    calls = len(worker.calls)
    duplicate = run.evaluate(ACTION)
    assert duplicate.execution_feedback is not None
    assert duplicate.execution_feedback.source_id == first.candidate_id
    assert len(worker.calls) == calls + 1  # Validation only, no extra executions.
    assert best_feedback_candidate(run) == first.candidate_id


def test_snapshot_isolated_from_later_model_timeout() -> None:
    worker = Worker(failure="candidate_execution", fail_execution=3)
    run = evaluator(
        worker,
        settings=SETTINGS.model_copy(
            update={
                "candidate_feedback_warmups": 1,
                "candidate_feedback_measurements": 1,
            }
        ),
    )
    run.start()
    run.evaluate(ACTION)
    selected, alternate = feedback_selection_evaluator(run)
    assert selected == "candidate-01"
    assert alternate.worker is worker
    assert isinstance(run.finish(random.Random(0)), TimedOutOutcome)
    model_bytes = run.record().model_dump_json()
    assert not alternate.candidates[0].execution_timed_out
    assert isinstance(alternate.finish(random.Random(1)), MeasuredOutcome)
    assert run.record().model_dump_json() == model_bytes
    assert alternate.execution_counts.initial_default == 0
    assert alternate.execution_counts.candidate_feedback == 0
    assert alternate.execution_counts.final_paired == 8
    with pytest.raises(RuntimeError, match="precede final"):
        feedback_selection_evaluator(run)


def scripted_choices(
    activity: EvaluationActivity,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_choice: str,
    failure: str | None,
) -> None:
    original_command = activity.command
    executions: dict[bool, int] = {False: 0, True: 0}

    def command(command: list[str], query: str) -> subprocess.CompletedProcess[str]:
        response = original_command(command, query)
        if "/*+" not in query:
            return response
        second = "Set(seq_page_cost 3)" in query
        if "ANALYZE" in query:
            executions[second] += 1
        final = executions[second] > 2
        if second and final and failure:
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "canceling statement due to statement timeout"
                if failure == "timeout"
                else "connection lost",
            )
        document = json.loads(response.stdout)
        document[0]["Execution Time"] = (
            (80.0 if final else 50.0) if second else (400.0 if final else 200.0)
        )
        return subprocess.CompletedProcess(
            command, 0, json.dumps(document), response.stderr
        )

    original_request = activity.request

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        response = original_request(transport, path, body)
        if path != "chat/completions":
            return response
        assert body is not None
        messages = body["messages"]
        assert isinstance(messages, list)
        turns = sum(isinstance(m, dict) and m.get("role") == "tool" for m in messages)
        choices = response["choices"]
        assert isinstance(choices, list) and isinstance(choices[0], dict)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        calls = message["tool_calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        if turns == 2:
            calls[0]["function"] = {
                "name": "evaluate_candidate",
                "arguments": json.dumps(
                    {"action": {"version": 1, "settings": {"seq_page_cost": 3.0}}}
                ),
            }
        elif turns == 3:
            calls[0]["function"] = {
                "name": "finish",
                "arguments": json.dumps({"selected_candidate_id": model_choice}),
            }
        return response

    monkeypatch.setattr(activity, "command", command)
    monkeypatch.setattr(HttpTransport, "request", request)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model_choice", ["candidate-01", "candidate-02", "default"])
def test_actual_evaluation_preserves_model_and_measures_rule(
    enabled: bool,
    model_choice: str,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_choices(activity, monkeypatch, model_choice=model_choice, failure=None)
    config = evaluation_config.model_copy(
        update={
            "agent": evaluation_config.agent.model_copy(
                update={"candidate_attempts": 2}
            ),
            "evaluation": evaluation_config.evaluation.model_copy(
                update={"selection_policy": "best_feedback" if enabled else "model"}
            ),
        }
    )
    output = tmp_path / "eval"
    report = run_evaluation(
        output, config, pool_config, postgres_config, benchmark_task_sets["job"]
    )
    (record,) = saved_rollouts(output)
    model_speedup = {"candidate-01": 0.25, "candidate-02": 1.25, "default": 1.0}[
        model_choice
    ]
    assert report.summary.performance.geometric_mean_speedup == pytest.approx(
        model_speedup
    )
    assert record.trace is not None
    assert record.trace.model_responses[-1].message.tool_calls is not None
    assert record.trace.model_responses[-1].message.tool_calls[
        0
    ].function.arguments == json.dumps({"selected_candidate_id": model_choice})
    assert len(record.trace.model_responses) == 4
    base_executions = 8 + (0 if model_choice == "default" else 8)
    if enabled:
        comparison = record.feedback_selection
        assert comparison is not None and comparison.rollout.final is not None
        assert comparison.model_selected_candidate_id == model_choice
        assert comparison.selected_candidate_id == "candidate-02"
        assert (
            comparison.rollout.final.speedup == 1.25
        )  # Fresh timing, not the 2x probe.
        assert comparison.reused_model_result == (model_choice == "candidate-02")
        summary = report.summary.feedback_selection
        assert summary is not None
        assert summary.performance.geometric_mean_speedup == 1.25
        assert summary.applied_rollout_count == 1
        assert summary.changed_selection_count == (model_choice != "candidate-02")
        assert summary.default_selection_count == 0
        extra = 0 if comparison.reused_model_result else 8
        assert summary.additional_execution_counts.final_paired == extra
        assert sum("ANALYZE" in q for q in activity.queries) == base_executions + extra
    else:
        assert record.feedback_selection is None
        assert "feedback_selection" not in record.model_dump()
        assert report.summary.feedback_selection is None
        assert sum("ANALYZE" in q for q in activity.queries) == base_executions
    assert not activity.claimed_workers


@pytest.mark.parametrize(
    "mode", ["keep", "invalid", "provider_failure_after_candidate"]
)
def test_default_fallback_and_failed_search_accounting(
    mode: str,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    from qorl.model.exceptions import ModelRequestError

    activity.mode = mode
    config = evaluation_config.model_copy(
        update={
            "evaluation": evaluation_config.evaluation.model_copy(
                update={"selection_policy": "best_feedback"}
            )
        }
    )
    output = tmp_path / "eval"
    if mode == "provider_failure_after_candidate":
        with pytest.raises(ModelRequestError):
            run_evaluation(
                output, config, pool_config, postgres_config, benchmark_task_sets["job"]
            )
    else:
        run_evaluation(
            output, config, pool_config, postgres_config, benchmark_task_sets["job"]
        )
    (record,) = saved_rollouts(output)
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    summary = report.summary.feedback_selection
    assert summary is not None
    if mode == "provider_failure_after_candidate":
        assert record.feedback_selection is None
        assert summary.applied_rollout_count == 0
        assert summary.performance.failure_count == 1
        assert summary.performance.scored_rollout_count == 0
    else:
        assert record.rollout.final is not None
        assert record.feedback_selection is not None
        assert record.feedback_selection.selected_candidate_id == "default"
        assert summary.performance.geometric_mean_speedup == 1.0
        assert summary.default_selection_count == 1
        assert summary.additional_execution_counts.final_paired == 0
        assert record.feedback_selection.reused_model_result == (mode == "keep")
        if mode == "invalid":
            assert record.rollout.final.kind == OutcomeKind.NO_VALID_CANDIDATE
            assert report.summary.valid_plan_rollout_count == 0
    assert not activity.claimed_workers


def test_same_choice_reuses_final_timeout_without_resampling(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_choices(
        activity, monkeypatch, model_choice="candidate-02", failure="timeout"
    )
    config = evaluation_config.model_copy(
        update={
            "agent": evaluation_config.agent.model_copy(
                update={"candidate_attempts": 2}
            ),
            "evaluation": evaluation_config.evaluation.model_copy(
                update={"selection_policy": "best_feedback"}
            ),
        }
    )
    output = tmp_path / "eval"
    report = run_evaluation(
        output, config, pool_config, postgres_config, benchmark_task_sets["job"]
    )
    (record,) = saved_rollouts(output)
    comparison = record.feedback_selection
    assert comparison is not None and comparison.reused_model_result
    assert comparison.selected_candidate_id == "candidate-02"
    assert comparison.rollout == record.rollout
    assert report.summary.performance.timeout_count == 1
    assert report.summary.feedback_selection is not None
    assert report.summary.feedback_selection.performance.timeout_count == 1
    assert (
        report.summary.feedback_selection.additional_execution_counts.final_paired == 0
    )
    assert sum("ANALYZE" in q for q in activity.queries) == 9


def test_configuration_requires_execution_feedback(
    evaluation_config: EvaluationExperimentConfig,
) -> None:
    document = evaluation_config.model_dump()
    document["evaluation"]["selection_policy"] = "best_feedback"
    document["measurement"]["candidate_feedback_measurements"] = 0
    document["measurement"]["candidate_feedback_warmups"] = 0
    with pytest.raises(ValueError, match="requires candidate execution feedback"):
        EvaluationExperimentConfig.model_validate(document)


@pytest.mark.parametrize("failure", ["timeout", "connection"])
def test_rule_measurement_failure_cannot_erase_model_evidence(
    failure: str,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_choices(
        activity, monkeypatch, model_choice="candidate-01", failure=failure
    )
    config = evaluation_config.model_copy(
        update={
            "agent": evaluation_config.agent.model_copy(
                update={"candidate_attempts": 2}
            ),
            "evaluation": evaluation_config.evaluation.model_copy(
                update={"selection_policy": "best_feedback"}
            ),
        }
    )
    output = tmp_path / "eval"
    if failure == "connection":
        with pytest.raises(RuntimeError, match="1 failed feedback-selection"):
            run_evaluation(
                output, config, pool_config, postgres_config, benchmark_task_sets["job"]
            )
    else:
        run_evaluation(
            output, config, pool_config, postgres_config, benchmark_task_sets["job"]
        )
    (record,) = saved_rollouts(output)
    assert record.rollout.final is not None and record.rollout.final.speedup == 0.25
    comparison = record.feedback_selection
    assert comparison is not None
    if failure == "timeout":
        assert comparison.rollout.final is not None
        assert comparison.rollout.final.kind == OutcomeKind.TIMED_OUT
        assert comparison.rollout.final.speedup is None
    else:
        assert comparison.rollout.failure is not None
        assert comparison.rollout.final is None
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert report.summary.performance.failure_count == 0
    assert report.status == (
        RunStatus.COMPLETED
        if failure == "timeout"
        else RunStatus.COMPLETED_WITH_FAILURES
    )
    assert not activity.claimed_workers
