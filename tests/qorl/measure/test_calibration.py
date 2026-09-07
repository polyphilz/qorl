from __future__ import annotations

import runpy
import sys
from collections import Counter
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
import tomli_w
from pydantic import TypeAdapter

from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CalibrationExperimentConfig,
    CreateRequest,
    ExperimentMethod,
    load_config,
)
from qorl.measure import calibration
from qorl.measure.schemas import (
    CalibratedTask,
    CalibrationReport,
    CalibrationSettings,
    CalibrationTaskResult,
    FailedCalibrationTask,
    RunStatus,
)
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import QueryTimeoutError
from qorl.postgres.schemas import ExplainResult, PostgresIndexes
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import PoolConfig, WorkerSlot

TASK_RESULT = TypeAdapter(CalibrationTaskResult)


@dataclass(frozen=True)
class Execution:
    worker: int
    sql: str
    timeout_ms: int


@dataclass
class PoolActivity:
    started: list[ContainerPool] = field(default_factory=list)
    closed: list[ContainerPool] = field(default_factory=list)
    captures: list[tuple[int, str]] = field(default_factory=list)
    executions: list[Execution] = field(default_factory=list)


def explain(elapsed_ms: float = 1.5, hits: int = 100) -> ExplainResult:
    return ExplainResult(
        document={
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": "title",
                "Plan Rows": 100,
                "Shared Hit Blocks": hits,
                "Shared Read Blocks": 5,
            },
            "Planning Time": 0.2,
            "Execution Time": elapsed_ms,
        },
        hint_diagnostics="",
    )


@pytest.fixture
def activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, postgres_indexes: PostgresIndexes
) -> PoolActivity:
    observed = PoolActivity()
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")
    monkeypatch.setattr(calibration, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(ContainerPool, "create", lambda pool: None)
    monkeypatch.setattr(ContainerPool, "restore", lambda pool, archive: None)
    monkeypatch.setattr(
        ContainerPool, "start", lambda pool: observed.started.append(pool)
    )
    monkeypatch.setattr(
        ContainerPool, "close", lambda pool: observed.closed.append(pool)
    )
    monkeypatch.setattr(PostgresClient, "read_indexes", lambda worker: postgres_indexes)

    def capture(
        pool: ContainerPool, slot: WorkerSlot, output: Path, phase: str
    ) -> None:
        observed.captures.append((slot.resources.index, phase))

    def execute(
        worker: PostgresClient,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        assert analyze and not hint
        slot = next(
            slot for slot in observed.started[0].workers if slot.client is worker
        )
        observed.executions.append(Execution(slot.resources.index, sql, timeout_ms))
        return explain()

    monkeypatch.setattr(calibration, "capture_environment", capture)
    monkeypatch.setattr(PostgresClient, "explain", execute)
    return observed


def selection(task_set: TaskSet, count: int = 1) -> TaskSelection:
    return TaskSelection(
        benchmark_id=BenchmarkId(task_set.task_set_id),
        task_ids=[task.task_id for task in task_set.tasks[:count]],
    )


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
@pytest.mark.parametrize(
    "config_id,worker_count",
    [("000-poolconf-1x32", 1), ("001-poolconf-2x16", 2), ("002-poolconf-4x8", 4)],
)
@pytest.mark.parametrize("max_warmup_runs,num_trials", [(5, 20), (2, 2), (3, 4)])
def test_calibration_measures_only_selected_tasks_and_records_applied_pool(
    repository_root: Path,
    tmp_path: Path,
    activity: PoolActivity,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    benchmark: str,
    config_id: str,
    worker_count: int,
    max_warmup_runs: int,
    num_trials: int,
) -> None:
    task_set = benchmark_task_sets[benchmark]
    selected = selection(task_set, 5)
    settings = CalibrationSettings(
        max_warmup_runs=max_warmup_runs,
        num_trials=num_trials,
        default_timeout_seconds=7,
    )
    output = tmp_path / "calibration"
    config = load_pool_config(
        repository_root / "docker/worker_pool/configs" / config_id
    )
    report = calibration.calibrate(
        task_set,
        selected,
        output,
        settings=settings,
        postgres_config=postgres_config,
        pool_config=config,
    )
    assert report == CalibrationReport.model_validate_json(
        (output / "calibration.json").read_bytes()
    )
    assert report.status == RunStatus.COMPLETED
    assert report.benchmark_id.value == benchmark
    assert report.completed_task_count == 5
    assert report.failed_task_count == 0
    assert report.measurement == settings
    assert report.statement_timeout_ms == 7_000
    assert report.worker_pool is not None
    assert report.worker_pool.worker_count == worker_count
    assert report.worker_pool.id == config_id
    assert report.worker_pool.postgres_config == postgres_config.manifest()
    assert len(activity.started) == 1
    assert activity.closed == activity.started
    assert activity.captures == [
        (slot, phase) for phase in ("pre", "post") for slot in range(worker_count)
    ]
    assert len(activity.executions) == 5 * (2 + num_trials)
    assert {item.timeout_ms for item in activity.executions} == {7_000}
    assert set((output / "tasks").glob("*.json")) == {
        output / "tasks" / f"{task_id}.json" for task_id in selected.task_ids
    }
    for task in task_set.resolve(selected):
        record = TASK_RESULT.validate_json(
            (output / "tasks" / f"{task.task_id}.json").read_bytes()
        )
        assert isinstance(record, CalibratedTask)
        assert len(record.warmups) == 2
        assert len(record.measurements) == num_trials
        assert record.summary.warmup_stable
        assert record.summary.measurement_count == num_trials
        assert record.summary.median_execution_time_ms == 1.5
        assert record.summary.coefficient_of_variation == 0
        assert record.summary.distinct_plan_count == 1
        assert record.representative_explain_analyze["Execution Time"] == 1.5
        observed = [
            item for item in activity.executions if item.sql == task_set.load_sql(task)
        ]
        assert len(observed) == 2 + num_trials
        assert {item.worker for item in observed} == {record.worker.slot}
        assert [item.run for item in record.measurements] == list(
            range(1, num_trials + 1)
        )


def test_summary_excludes_warmups_and_uses_sample_variation(
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    timings = [100, 200, 10, 30]
    executor = Mock(side_effect=[explain(value) for value in timings])
    monkeypatch.setattr(PostgresClient, "explain", executor)
    task_set = benchmark_task_sets["job"]
    selected = selection(task_set)
    calibration.calibrate(
        task_set,
        selected,
        tmp_path / "calibration",
        settings=CalibrationSettings(
            max_warmup_runs=5, num_trials=2, default_timeout_seconds=1
        ),
        postgres_config=postgres_config,
        pool_config=pool_config,
    )
    record = TASK_RESULT.validate_json(
        (tmp_path / "calibration/tasks" / f"{selected.task_ids[0]}.json").read_bytes()
    )
    assert isinstance(record, CalibratedTask)
    assert [item.execution_time_ms for item in record.warmups] == timings[:2]
    assert record.summary.median_execution_time_ms == 20
    assert record.summary.sample_standard_deviation_ms == pytest.approx(14.1421356237)
    assert record.summary.coefficient_of_variation == pytest.approx(0.7071067812)


def test_statement_timeout_retains_partial_evidence_and_continues_other_tasks(
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    task_set = benchmark_task_sets["ceb"]
    selected = selection(task_set, 2)
    failed_sql = task_set.load_sql(task_set.resolve(selected)[0])
    calls: Counter[str] = Counter()

    def execute(
        worker: PostgresClient, sql: str, timeout_ms: int, *, analyze: bool = False
    ) -> ExplainResult:
        assert timeout_ms == 10
        calls[sql] += 1
        if sql == failed_sql and calls[sql] == 4:
            raise QueryTimeoutError(timeout_ms)
        return explain()

    monkeypatch.setattr(PostgresClient, "explain", execute)
    output = tmp_path / "calibration"
    with pytest.raises(RuntimeError, match="1 failed tasks"):
        calibration.calibrate(
            task_set,
            selected,
            output,
            settings=CalibrationSettings(
                max_warmup_runs=5, num_trials=2, default_timeout_seconds=0.01
            ),
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
    report = CalibrationReport.model_validate_json(
        (output / "calibration.json").read_bytes()
    )
    assert report.status == RunStatus.COMPLETED_WITH_FAILURES
    assert report.completed_task_count == report.failed_task_count == 1
    failed = TASK_RESULT.validate_json(
        (output / "tasks" / f"{selected.task_ids[0]}.json").read_bytes()
    )
    assert isinstance(failed, FailedCalibrationTask)
    assert len(failed.warmups) == 2
    assert len(failed.measurements) == 1
    assert failed.statement_timeout_ms == 10
    assert failed.error_type == "QueryTimeoutError"
    assert activity.closed == activity.started


@pytest.mark.parametrize(
    "failure", [ContainerError("startup failed"), KeyboardInterrupt()]
)
def test_startup_failure_closes_pool_and_records_failure(
    failure: BaseException,
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    def fail(pool: ContainerPool) -> None:
        activity.started.append(pool)
        raise failure

    monkeypatch.setattr(ContainerPool, "start", fail)
    task_set = benchmark_task_sets["job"]
    with pytest.raises(type(failure)):
        calibration.calibrate(
            task_set,
            selection(task_set),
            tmp_path / "calibration",
            settings=CalibrationSettings(
                max_warmup_runs=2, num_trials=2, default_timeout_seconds=1
            ),
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
    report = CalibrationReport.model_validate_json(
        (tmp_path / "calibration/calibration.json").read_bytes()
    )
    assert report.status == (
        RunStatus.INTERRUPTED
        if isinstance(failure, KeyboardInterrupt)
        else RunStatus.FAILED
    )
    assert report.error
    assert activity.closed == activity.started
    assert not activity.executions


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_environment_capture_failure_still_closes_pool(
    phase: str,
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    def capture(
        pool: ContainerPool, slot: WorkerSlot, output: Path, actual_phase: str
    ) -> None:
        if actual_phase == phase:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(calibration, "capture_environment", capture)
    task_set = benchmark_task_sets["job"]
    with pytest.raises(RuntimeError, match="capture failed"):
        calibration.calibrate(
            task_set,
            selection(task_set),
            tmp_path / "calibration",
            settings=CalibrationSettings(
                max_warmup_runs=2, num_trials=2, default_timeout_seconds=1
            ),
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
    assert activity.closed == activity.started
    report = CalibrationReport.model_validate_json(
        (tmp_path / "calibration/calibration.json").read_bytes()
    )
    assert report.status == RunStatus.FAILED
    assert report.completed_task_count == (1 if phase == "post" else 0)


def test_cancelled_task_does_not_execute_sql(
    activity: PoolActivity,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    task_set = benchmark_task_sets["job"]
    pool = ContainerPool("cancelled", pool_config, postgres_config)
    stop = Event()
    stop.set()
    with pytest.raises(CancelledError):
        calibration.calibrate_task(
            pool,
            task_set,
            task_set.tasks[0],
            CalibrationSettings(
                max_warmup_runs=2, num_trials=2, default_timeout_seconds=1
            ),
            stop,
        )
    assert not activity.executions
    with pool.claim_worker() as slot:
        assert slot in pool.workers


@pytest.mark.parametrize("timeout", [False, True])
def test_generated_entrypoint_measures_saved_selection_and_retains_results(
    timeout: bool,
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", tmp_path / "experiments")
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", tmp_path / "outputs")
    experiment = create.create_experiment(
        CreateRequest(
            name="small-calibration",
            method=ExperimentMethod.CALIBRATE,
            tasksets=("test=ceb[2a:2]",),
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
        )
    )
    config = load_config(experiment / "config.toml")
    assert isinstance(config, CalibrationExperimentConfig)
    config = config.model_copy(
        update={
            "measurement": CalibrationSettings(
                max_warmup_runs=2, num_trials=2, default_timeout_seconds=0.01
            )
        }
    )
    (experiment / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    selected = run.load_inputs(experiment).selections[TaskRole.TEST]
    task_set = benchmark_task_sets["ceb"]
    first_sql = task_set.load_sql(task_set.resolve(selected)[0])
    execute = PostgresClient.explain

    def explain_or_timeout(
        worker: PostgresClient, sql: str, timeout_ms: int, *, analyze: bool = False
    ) -> ExplainResult:
        result = execute(worker, sql, timeout_ms, analyze=analyze)
        if timeout and sql == first_sql:
            raise QueryTimeoutError(timeout_ms)
        return result

    monkeypatch.setattr(PostgresClient, "explain", explain_or_timeout)
    monkeypatch.setattr(
        sys, "argv", [str(experiment / "run.py"), "--stage", "calibrate"]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(experiment / "run.py"), run_name="__main__")
    assert error.value.code == (1 if timeout else 0)
    output = run.OUTPUTS_DIRECTORY / experiment.name / "000"
    assert run.load_inputs(output) == run.load_inputs(experiment)
    report = CalibrationReport.model_validate_json(
        (output / "calibration/calibration.json").read_bytes()
    )
    assert report.status == (
        RunStatus.COMPLETED_WITH_FAILURES if timeout else RunStatus.COMPLETED
    )
    assert report.completed_task_count == (1 if timeout else 2)
    assert report.failed_task_count == (1 if timeout else 0)
    assert {item.sql for item in activity.executions} == {
        task_set.load_sql(task) for task in task_set.resolve(selected)
    }
    assert len(activity.executions) == (5 if timeout else 8)
    assert {item.timeout_ms for item in activity.executions} == {10}
    assert activity.closed == activity.started


@pytest.mark.parametrize("zero_runtime", [False, True])
def test_unstable_warmups_stop_at_cap_without_discarding_measurements(
    zero_runtime: bool,
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    responses = [
        explain(elapsed_ms=0 if zero_runtime else 10, hits=hits)
        for hits in [10, 20, 30, 40, 50, 50, 50]
    ]
    executor = Mock(side_effect=responses)
    monkeypatch.setattr(PostgresClient, "explain", executor)
    task_set = benchmark_task_sets["job"]
    selected = selection(task_set)
    output = tmp_path / "calibration"
    calibration.calibrate(
        task_set,
        selected,
        output,
        settings=CalibrationSettings(
            max_warmup_runs=5, num_trials=2, default_timeout_seconds=1
        ),
        postgres_config=postgres_config,
        pool_config=pool_config,
    )
    record = TASK_RESULT.validate_json(
        (output / "tasks" / f"{selected.task_ids[0]}.json").read_bytes()
    )
    assert isinstance(record, CalibratedTask)
    assert not record.summary.warmup_stable
    assert len(record.warmups) == 5
    assert len(record.measurements) == 2
    assert record.summary.coefficient_of_variation == (None if zero_runtime else 0)
    assert executor.call_count == 7


@pytest.mark.parametrize(
    "failure", [RuntimeError("unexpected SQL error"), KeyboardInterrupt()]
)
def test_unhandled_query_failure_records_failure_and_closes_pool(
    failure: BaseException,
    tmp_path: Path,
    activity: PoolActivity,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    execute = Mock(side_effect=failure)
    monkeypatch.setattr(PostgresClient, "explain", execute)
    task_set = benchmark_task_sets["job"]
    with pytest.raises(type(failure)):
        calibration.calibrate(
            task_set,
            selection(task_set, 20),
            tmp_path / "calibration",
            settings=CalibrationSettings(
                max_warmup_runs=2, num_trials=2, default_timeout_seconds=1
            ),
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
    report = CalibrationReport.model_validate_json(
        (tmp_path / "calibration/calibration.json").read_bytes()
    )
    assert report.status == (
        RunStatus.INTERRUPTED
        if isinstance(failure, KeyboardInterrupt)
        else RunStatus.FAILED
    )
    assert activity.closed == activity.started
    assert execute.call_count > 0
