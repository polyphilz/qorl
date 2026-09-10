import runpy
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import tomli_w
from tests.qorl.evaluation.conftest import EvaluationActivity

from qorl.agent.types import StopReason
from qorl.evaluation import evaluate
from qorl.evaluation.evaluate import summarize_performance
from qorl.evaluation.schemas import (
    EvaluationReport,
    EvaluationRollout,
    EvaluationSettings,
)
from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CreateRequest,
    EvaluationExperimentConfig,
    ExperimentMethod,
    ModelExperimentConfig,
    RunRequest,
    RunStage,
    load_config,
)
from qorl.measure.schemas import (
    NoValidCandidateOutcome,
    OutcomeKind,
    PairedMeasurements,
    RolloutFailure,
    RolloutRecord,
    RunStatus,
)
from qorl.model.client import HttpTransport
from qorl.model.exceptions import ModelRequestError, ModelResponseError
from qorl.model.schemas import (
    AstraInferenceSettings,
    JsonObject,
    ModelPreset,
    ModelProvider,
    ModelSettings,
    ReasoningEffort,
    TokenUsage,
)
from qorl.postgres.config import PostgresConfig
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig


def run_evaluation(
    output: Path,
    config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    task_set: TaskSet,
    *,
    count: int = 1,
    rollouts: int = 1,
) -> EvaluationReport:
    return evaluate.evaluate(
        task_set,
        TaskSelection(
            benchmark_id=BenchmarkId(task_set.task_set_id),
            task_ids=[task.task_id for task in task_set.tasks[:count]],
        ),
        output,
        split=TaskRole.TEST,
        model=config.model,
        inference=config.inference,
        serving_gpu_ids=config.resources.serving_gpu_ids if config.resources else None,
        agent=config.agent,
        measurement=config.measurement,
        settings=EvaluationSettings(rollouts_per_task=rollouts),
        seed=config.experiment.seed,
        postgres_config=postgres_config,
        pool_config=pool_config,
    )


def saved_rollouts(output: Path) -> list[EvaluationRollout]:
    return [
        EvaluationRollout.model_validate_json(path.read_bytes())
        for path in sorted((output / "rollouts").glob("*/*.json"))
    ]


@pytest.mark.parametrize(
    "failure", [None, "budget", "missing_usage", "malformed_choices", "invalid_usage"]
)
def test_openrouter_dispatch_and_paid_failure_evidence(
    failure: str | None,
    openrouter_preset: ModelPreset,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HttpTransport.request
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        assert path == "chat/completions"
        raw = original(transport, path, body)
        raw["model"] = openrouter_preset.model.name_or_path
        raw["provider"] = "scripted"
        if failure == "budget":
            raw["usage"] = {"prompt_tokens": 999999, "completion_tokens": 5}
        elif failure == "missing_usage":
            raw["usage"] = {"completion_tokens": 5}
        elif failure == "malformed_choices":
            raw["choices"] = []
            raw["usage"] = {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 20},
            }
        elif failure == "invalid_usage":
            raw["choices"] = []
            raw["usage"] = {"prompt_tokens": -1, "completion_tokens": 5}
        return raw

    monkeypatch.setattr(HttpTransport, "request", request)
    config = evaluation_config.model_copy(
        update={
            "model": openrouter_preset.model,
            "inference": openrouter_preset.inference,
            "resources": None,
        }
    )
    output = tmp_path / "openrouter"
    if failure:
        with pytest.raises(ModelResponseError):
            run_evaluation(
                output, config, pool_config, postgres_config, benchmark_task_sets["job"]
            )
        report = EvaluationReport.model_validate_json(
            (output / "evaluation.json").read_bytes()
        )
    else:
        report = run_evaluation(
            output, config, pool_config, postgres_config, benchmark_task_sets["job"]
        )
    assert not activity.served
    (record,) = saved_rollouts(output)
    assert record.trace is not None
    if failure:
        assert report.summary.performance.failure_count == 1
        assert record.rollout.final is None and record.rollout.failure is not None
        assert not record.trace.tool_events and not record.rollout.candidates
        assert not record.trace.model_responses
        (evidence,) = record.trace.model_failures
        assert evidence.request == activity.requests[0]
        assert evidence.raw_response["provider"] == "scripted"
        assert record.trace.usage == evidence.usage
        assert report.summary.usage == evidence.usage
        if failure == "invalid_usage":
            assert evidence.usage == TokenUsage()
        elif failure == "malformed_choices":
            assert evidence.usage == TokenUsage(
                prompt_tokens=100,
                completion_tokens=5,
                reasoning_tokens=3,
                cached_tokens=20,
            )
        else:
            assert record.trace.usage.completion_tokens == 5
            assert record.trace.usage.prompt_tokens == (
                999999 if failure == "budget" else None
            )
        assert sum("ANALYZE" in query for query in activity.queries) == 4
    else:
        assert report.summary.performance.failure_count == 0
        assert record.rollout.final is not None
        assert record.rollout.final.kind == OutcomeKind.MEASURED
        assert len(record.trace.model_responses) == 3
        assert "model_failures" not in record.trace.model_dump()
    assert activity.closed == activity.pools and not activity.claimed_workers


def test_openrouter_indexed_multiple_calls_reach_harness(
    openrouter_preset: ModelPreset,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HttpTransport.request
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        raw = original(transport, path, body)
        raw["model"] = openrouter_preset.model.name_or_path
        choices = raw["choices"]
        assert isinstance(choices, list) and isinstance(choices[0], dict)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        calls = message["tool_calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        calls[0]["index"] = 0
        if len(activity.requests) == 1:
            calls.append(
                {
                    "id": "extra-call",
                    "index": 1,
                    "type": "function",
                    "function": {
                        "name": "evaluate_candidate",
                        "arguments": '{ "action": {"version": 1} }',
                    },
                }
            )
        else:
            calls[0]["function"] = {"name": "keep_default", "arguments": "{}"}
        return raw

    monkeypatch.setattr(HttpTransport, "request", request)
    config = evaluation_config.model_copy(
        update={
            "model": openrouter_preset.model,
            "inference": openrouter_preset.inference,
            "resources": None,
        }
    )
    output = tmp_path / "indexed"
    report = run_evaluation(
        output, config, pool_config, postgres_config, benchmark_task_sets["job"]
    )
    (record,) = saved_rollouts(output)
    assert (
        report.status == RunStatus.COMPLETED
        and report.summary.performance.failure_count == 0
    )
    assert record.trace is not None and not record.trace.model_failures
    calls = record.trace.model_responses[0].message.tool_calls
    assert calls is not None and [call.id for call in calls] == ["call-0", "extra-call"]
    assert [event.tool_call_id for event in record.trace.tool_events] == [
        "call-0",
        "extra-call",
        "call-2",
    ]
    assert record.trace.tool_events[1].result["error"] == "call one tool at a time"
    assert not record.rollout.candidates
    assert (
        record.rollout.final is not None
        and record.rollout.final.kind == OutcomeKind.KEPT_DEFAULT
    )
    assert sum("ANALYZE" in query for query in activity.queries) == 4
    assert activity.closed == activity.pools and not activity.claimed_workers


@pytest.mark.parametrize("empty", [True, False])
def test_openrouter_output_limit_does_not_abort_scheduled_rollout(
    empty: bool,
    openrouter_preset: ModelPreset,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HttpTransport.request
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        raw = original(transport, path, body)
        raw["model"] = openrouter_preset.model.name_or_path
        if len(activity.requests) == 1:
            choices = raw["choices"]
            assert isinstance(choices, list) and isinstance(choices[0], dict)
            choices[0]["finish_reason"] = "length"
            if empty:
                choices[0]["message"] = {"role": "assistant", "content": None}
        return raw

    monkeypatch.setattr(HttpTransport, "request", request)
    config = evaluation_config.model_copy(
        update={
            "model": openrouter_preset.model.model_copy(
                update={"max_concurrent_requests": 1}
            ),
            "inference": openrouter_preset.inference,
            "resources": None,
        }
    )
    output = tmp_path / "output-limit"
    report = run_evaluation(
        output,
        config,
        replace(pool_config, workers=pool_config.workers[:1]),
        postgres_config,
        benchmark_task_sets["job"],
        rollouts=2,
    )
    first, second = saved_rollouts(output)
    assert report.status == RunStatus.COMPLETED
    assert report.summary.performance.failure_count == 0
    assert report.summary.stop_reason_counts[StopReason.MODEL_OUTPUT_LIMIT] == 1
    assert first.trace is not None and second.trace is not None
    assert first.trace.stop_reason == StopReason.MODEL_OUTPUT_LIMIT
    assert first.trace.model_responses[0].truncated
    assert not first.trace.tool_events and not first.trace.model_failures
    assert not first.rollout.candidates and first.rollout.failure is None
    assert (
        first.rollout.final is not None
        and first.rollout.final.kind == OutcomeKind.NO_VALID_CANDIDATE
    )
    assert (
        second.rollout.final is not None
        and second.rollout.final.kind == OutcomeKind.MEASURED
    )
    assert len(second.trace.tool_events) == 3
    assert sum("ANALYZE" in query for query in activity.queries) == 18
    assert activity.closed == activity.pools and not activity.claimed_workers


@pytest.mark.parametrize(
    "mode,kind,executions,valid,novel",
    [
        ("measured", OutcomeKind.MEASURED, 14, 1, 1),
        ("non_novel", OutcomeKind.MEASURED, 14, 1, 0),
        ("duplicate", OutcomeKind.DEFAULT_DUPLICATE, 4, 1, 0),
        ("keep", OutcomeKind.KEPT_DEFAULT, 4, 0, 0),
        ("invalid", OutcomeKind.NO_VALID_CANDIDATE, 4, 0, 0),
        ("candidate_timeout", OutcomeKind.TIMED_OUT, 4, 0, 0),
        ("truncated", OutcomeKind.NO_VALID_CANDIDATE, 4, 0, 0),
    ],
)
def test_actual_conversation_measurements_and_reporting(
    mode: str,
    kind: OutcomeKind,
    executions: int,
    valid: int,
    novel: int,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    activity.mode = mode
    output = tmp_path / "evaluation"
    report = run_evaluation(
        output,
        evaluation_config,
        pool_config,
        postgres_config,
        benchmark_task_sets["job"],
    )
    assert report.status == RunStatus.COMPLETED
    assert report.worker_pool is not None
    assert report.worker_pool.postgres_config == postgres_config.manifest()
    assert report.local_server is not None
    assert report.local_server.vllm_version == "test-runtime"
    assert report.summary.outcome_counts[kind] == 1
    assert report.summary.valid_plan_rate == valid
    assert report.summary.novel_plan_rate == novel
    assert report.summary.unrecorded_rollout_count == 0
    assert (
        sum(worker.client.explain_analyze_calls for worker in activity.pools[0].workers)
        == executions
    )
    assert activity.closed == activity.pools and activity.server_closed
    assert len(activity.captures) == len(pool_config.workers) * 2
    (record,) = saved_rollouts(output)
    assert record.rollout.final is not None and record.rollout.final.kind == kind
    assert record.trace is not None
    assert record.worker is not None
    if kind == OutcomeKind.MEASURED:
        assert report.summary.performance.geometric_mean_speedup == pytest.approx(20.0)
        assert [event.name for event in record.trace.tool_events] == [
            "get_plan",
            "evaluate_candidate",
            "finish",
        ]
        assert (
            record.rollout.final.selected_candidate_id
            == record.rollout.candidates[0].candidate_id
        )
    assert (
        EvaluationReport.model_validate_json((output / "evaluation.json").read_bytes())
        == report
    )


def test_queries_run_concurrently_and_seeds_do_not_depend_on_output_names(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    capsys: pytest.CaptureFixture[str],
) -> None:
    activity.barrier = Barrier(len(pool_config.workers))
    for name in ("first", "second"):
        report = run_evaluation(
            tmp_path / name,
            evaluation_config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
            count=2,
            rollouts=4,
        )
        assert report.summary.recorded_rollout_count == 8
        assert report.summary.recorded_task_count == 2
        assert (
            report.summary.distinct_novel_plan_count == 2
        )  # Same structure on each task.
        assert report.summary.distinct_novel_plan_yield == 2 / 8
        assert report.summary.novel_plan_rate == 1
    first, second = (
        saved_rollouts(tmp_path / "first"),
        saved_rollouts(tmp_path / "second"),
    )
    console = capsys.readouterr().out
    for record in first:
        label = f"{record.rollout.task_id} rollout={record.rollout_index}"
        assert f"[{label}] turn-01: get_plan" in console
        assert f"[{label}] candidate-01: validated" in console
    assert [(record.model_seed, record.measurement_seed) for record in first] == [
        (record.model_seed, record.measurement_seed) for record in second
    ]
    assert len({record.model_seed for record in first}) == 8
    assert {record.model_seed for record in first}.isdisjoint(
        record.measurement_seed for record in first
    )
    assert [record.rollout.final for record in first] == [
        record.rollout.final for record in second
    ]
    assert activity.closed == activity.pools


def test_astra_uses_the_same_harness_without_local_serving(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QORL_TEST_API_KEY", "test-secret")
    config = evaluation_config.model_copy(
        update={
            "model": evaluation_config.model.model_copy(
                update={
                    "provider": ModelProvider.OPENAI,
                    "revision": None,
                    "name_or_path": "gpt-6-astra",
                    "api_key_env": "QORL_TEST_API_KEY",
                    "base_url": "https://api.example/v1",
                }
            ),
            "inference": AstraInferenceSettings(
                max_tokens=2048, reasoning_effort=ReasoningEffort.LOW
            ),
            "resources": None,
        }
    )
    output = tmp_path / "astra"
    report = run_evaluation(
        output, config, pool_config, postgres_config, benchmark_task_sets["job"]
    )
    assert report.model == config.model and report.local_server is None
    assert not activity.served
    (record,) = saved_rollouts(output)
    assert record.trace is not None
    assert all(
        response.message.continuation is not None
        for response in record.trace.model_responses
    )
    assert all(
        "seed" not in response.request for response in record.trace.model_responses
    )
    assert report.summary.performance.geometric_mean_speedup == pytest.approx(20.0)
    assert "test-secret" not in (output / "evaluation.json").read_text()


@pytest.mark.parametrize(
    "mode,exception",
    [
        ("default_timeout", RuntimeError),
        ("provider_failure", ModelRequestError),
        ("provider_failure_after_candidate", ModelRequestError),
        ("interrupt", KeyboardInterrupt),
    ],
)
def test_partial_failures_are_saved_and_resources_are_closed(
    mode: str,
    exception: type[BaseException],
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    activity.mode = mode
    output = tmp_path / "failed"
    with pytest.raises(exception):
        run_evaluation(
            output,
            evaluation_config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
        )
    (record,) = saved_rollouts(output)
    assert record.rollout.failure is not None and record.rollout.final is None
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert report.summary.performance.failure_count == 1
    assert report.summary.performance.scored_rollout_count == 0
    assert report.summary.performance.geometric_mean_speedup is None
    assert report.status == (
        RunStatus.INTERRUPTED
        if mode == "interrupt"
        else RunStatus.COMPLETED_WITH_FAILURES
        if mode == "default_timeout"
        else RunStatus.FAILED
    )
    if mode == "provider_failure_after_candidate":
        assert record.rollout.candidates[0].constraints_satisfied
        assert report.summary.valid_plan_rate == 1
        assert record.trace is not None and len(record.trace.tool_events) == 2
    assert activity.closed == activity.pools and activity.server_closed


def test_completed_results_survive_main_thread_interruption_and_cleanup_wait(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_completed = evaluate.as_completed
    original_wait = evaluate.wait
    interrupted = False

    def interrupt_after_completion(
        fs: list[Future[EvaluationRollout]],
    ) -> Iterator[Future[EvaluationRollout]]:
        for future in original_completed(fs):
            yield future
            raise KeyboardInterrupt()

    def interrupt_wait_once(
        fs: list[Future[EvaluationRollout]],
    ) -> tuple[set[Future[EvaluationRollout]], set[Future[EvaluationRollout]]]:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original_wait(fs)

    monkeypatch.setattr(evaluate, "as_completed", interrupt_after_completion)
    monkeypatch.setattr(evaluate, "wait", interrupt_wait_once)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_evaluation(
            output,
            evaluation_config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
            count=2,
            rollouts=4,
        )
    records = saved_rollouts(output)
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert interrupted and report.status == RunStatus.INTERRUPTED
    assert report.summary.recorded_rollout_count == len(records) >= 1
    assert report.summary.unrecorded_rollout_count == 8 - len(records)
    assert any(record.rollout.final is not None for record in records)
    assert report.summary == evaluate.summarize_evaluation(records, 8)
    assert activity.closed == activity.pools and activity.server_closed


def test_preexisting_outputs_are_not_overwritten(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    output = tmp_path / "evaluation"
    run_evaluation(
        output,
        evaluation_config,
        pool_config,
        postgres_config,
        benchmark_task_sets["job"],
    )
    original = (output / "evaluation.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_evaluation(
            output,
            evaluation_config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
        )
    assert (output / "evaluation.json").read_bytes() == original
    assert len(activity.pools) == 1


def test_one_query_failure_does_not_discard_other_results(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = activity.command
    task_set = benchmark_task_sets["job"]
    bad_sql = task_set.load_sql(task_set.tasks[0]).strip()

    def command(args: list[str], query: str) -> subprocess.CompletedProcess[str]:
        if bad_sql in query:
            raise evaluate.PostgresError("query unavailable")
        return original(args, query)

    monkeypatch.setattr(activity, "command", command)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="1 failed rollouts"):
        run_evaluation(
            output, evaluation_config, pool_config, postgres_config, task_set, count=2
        )
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert report.summary.recorded_rollout_count == 2
    assert report.summary.performance.failure_count == 1
    assert report.summary.valid_plan_rate == 0.5
    assert report.summary.novel_plan_rate == 0.5
    assert report.summary.performance.scored_rollout_count == 1


def test_provider_failure_stops_queued_rollouts(
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    activity.mode = "provider_failure"
    single = replace(pool_config, workers=pool_config.workers[:1])
    output = tmp_path / "fatal"
    with pytest.raises(ModelRequestError, match="provider unavailable"):
        run_evaluation(
            output,
            evaluation_config,
            single,
            postgres_config,
            benchmark_task_sets["job"],
            count=20,
            rollouts=4,
        )
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert 1 <= report.summary.recorded_rollout_count < 80
    assert report.summary.unrecorded_rollout_count > 0
    assert report.error == "provider unavailable"
    assert (
        report.summary.performance.failure_count
        == report.summary.recorded_rollout_count
    )


@pytest.mark.parametrize(
    "method", [ExperimentMethod.EVAL, ExperimentMethod.SFT, ExperimentMethod.RL]
)
@pytest.mark.parametrize("attempts", [1, 5])
def test_created_entrypoint_reaches_evaluation_with_explicit_model_and_split(
    method: ExperimentMethod,
    attempts: int,
    tmp_path: Path,
    activity: EvaluationActivity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", tmp_path / "experiments")
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", tmp_path / "outputs")
    training = method != ExperimentMethod.EVAL
    directory = create.create_experiment(
        CreateRequest(
            name="evaluation-path",
            method=method,
            base_model_name_or_path="example/base",
            base_model_revision="a" * 40,
            tasksets=("train=ceb[2a:1]", "validation=ceb[4a:1]", "test=job")
            if training
            else ("test=ceb[2a:1]",),
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, ModelExperimentConfig)
    # Standalone evaluation uses the configured adapter; training evaluation uses
    # the explicit stage checkpoint. The serving boundary records what it receives.
    adapter = tmp_path / "explicit-adapter"
    if training:
        from qorl.rl import train as rl_training
        from qorl.sft import train as sft_training

        def exported_model(
            model: ModelSettings, training_directory: Path, checkpoint: Path
        ) -> ModelSettings:
            assert (
                training_directory
                == run.OUTPUTS_DIRECTORY / directory.name / "000/training"
            )
            return model.model_copy(update={"adapter_path": checkpoint})

        monkeypatch.setattr(
            rl_training if method == ExperimentMethod.RL else sft_training,
            "checkpoint_model",
            exported_model,
        )
    changed = config.model_copy(
        update={
            "agent": config.agent.model_copy(update={"candidate_attempts": attempts}),
            "evaluation": EvaluationSettings(rollouts_per_task=1),
            "model": config.model.model_copy(
                update={
                    "context_length": 20_480,
                    **({} if training else {"adapter_path": adapter}),
                }
            ),
            "inference": config.inference.model_copy(update={"max_tokens": 2048}),
        }
    )
    (directory / "config.toml").write_text(
        tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True))
    )
    role = TaskRole.VALIDATION if training else TaskRole.TEST
    arguments = [
        str(directory / "run.py"),
        "--stage",
        "evaluate",
        "--split",
        role.value,
    ]
    if training:
        run.create_run(directory, run.load_inputs(directory))
        arguments.extend(["--run", "000", "--checkpoint", str(adapter)])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(directory / "run.py"), run_name="__main__")
    assert exited.value.code == 0
    output = run.OUTPUTS_DIRECTORY / directory.name / "000"
    assert run.load_inputs(output) == run.load_inputs(directory)
    evaluation_dir = output / "evaluation" / role.value / "000"
    (record,) = saved_rollouts(evaluation_dir)
    selected = run.load_inputs(output).selections[role]
    assert record.rollout.task_id == selected.task_ids[0]
    assert record.trace is not None
    assert record.trace.selection.selected_candidate_id == "candidate-01"
    assert activity.served[0].adapter_path == adapter
    assert activity.served[0].name_or_path == "example/base"
    assert activity.served[0].revision == "a" * 40
    report_bytes = (evaluation_dir / "evaluation.json").read_bytes()
    report = EvaluationReport.model_validate_json(report_bytes)
    assert report.model.adapter_path == adapter
    request = RunRequest(
        RunStage.EVALUATE,
        number=0,
        split=role,
        checkpoint=adapter if training else None,
    )
    with pytest.raises(ValueError, match="cannot resume"):
        run.run_experiment(directory, replace(request, resume=True))
    run.run_experiment(directory, request)
    assert (output / "evaluation" / role.value / "001/evaluation.json").is_file()
    assert (evaluation_dir / "evaluation.json").read_bytes() == report_bytes
    if training:
        run.run_experiment(
            directory, replace(request, checkpoint=tmp_path / "different-adapter")
        )
        different = EvaluationReport.model_validate_json(
            (output / "evaluation" / role.value / "002/evaluation.json").read_bytes()
        )
        assert different.model.adapter_path == tmp_path / "different-adapter"
    changed = changed.model_copy(
        update={"experiment": changed.experiment.model_copy(update={"seed": 7})}
    )
    (directory / "config.toml").write_text(
        tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True))
    )
    with pytest.raises(ValueError, match="inputs changed"):
        run.run_experiment(directory, request)


@pytest.mark.parametrize("phase", ["server", "pool", "capture", "cleanup"])
def test_setup_and_cleanup_errors_leave_a_report_and_close_owned_resources(
    phase: str,
    tmp_path: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    if phase == "server":
        monkeypatch.setattr(
            evaluate,
            "serve_local_model",
            Mock(side_effect=RuntimeError("setup failed")),
        )
    elif phase == "pool":
        monkeypatch.setattr(
            evaluate, "start_pool", Mock(side_effect=RuntimeError("setup failed"))
        )
    elif phase == "capture":
        monkeypatch.setattr(
            evaluate,
            "capture_environment",
            Mock(side_effect=RuntimeError("setup failed")),
        )
    else:
        monkeypatch.setattr(
            evaluate.ContainerPool,
            "close",
            Mock(side_effect=RuntimeError("cleanup failed")),
        )
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="failed"):
        run_evaluation(
            output,
            evaluation_config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
        )
    report = EvaluationReport.model_validate_json(
        (output / "evaluation.json").read_bytes()
    )
    assert report.status == RunStatus.FAILED
    assert report.summary.recorded_rollout_count == (1 if phase == "cleanup" else 0)
    assert report.summary.unrecorded_rollout_count == (0 if phase == "cleanup" else 1)
    if phase != "server":
        assert activity.server_closed
    if phase == "capture":
        assert activity.closed == activity.pools


def test_summary_includes_failures_in_its_denominator(
    rollout_record: RolloutRecord,
) -> None:
    invalid = rollout_record.model_copy(
        update={"candidates": [], "final": NoValidCandidateOutcome()}
    )
    failed = rollout_record.model_copy(
        update={
            "final": None,
            "failure": RolloutFailure(
                operation="model",
                error_type="ModelError",
                error="unavailable",
                paired=PairedMeasurements(),
            ),
        }
    )
    summary = summarize_performance([rollout_record, invalid, failed])
    assert summary.rollout_count == 3
    assert summary.scored_rollout_count == 1
    assert summary.failure_count == 1
    assert summary.no_valid_candidate_count == 1
    assert summary.geometric_mean_speedup == pytest.approx(2.0)
    assert summary.candidate_workload_time_ms == 5.0
    assert summary.default_workload_time_ms == 10.0
    assert summary.total_workload_speedup == 2.0


def test_empty_summary_has_no_invented_speedup() -> None:
    summary = summarize_performance([])
    assert summary.rollout_count == 0
    assert summary.scored_rollout_count == 0
    assert summary.geometric_mean_speedup is None
    assert summary.total_workload_speedup is None
