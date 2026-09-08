from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from qorl.measure.schemas import (
    CalibratedTask,
    CalibrationReport,
    CalibrationSettings,
    CalibrationSummary,
    CalibrationTaskResult,
    FailedCalibrationTask,
    QueryObservation,
    RunStatus,
)
from qorl.taskset.schemas import BenchmarkId
from qorl.worker_pool.schemas import PoolConfig, WorkerManifest
from scripts.analysis.analyze_calibration import (
    analyze_calibration,
    compare_calibrations,
    estimate_noop_error,
    main,
    percentile,
)
from scripts.analysis.schemas import AnalysisReport, AnalysisSettings


@pytest.fixture
def worker(pool_config: PoolConfig) -> WorkerManifest:
    return pool_config.workers[0].manifest()


def task_record(
    worker: WorkerManifest,
    times: list[float],
    *,
    task_id: str = "job-01a",
    plan: str = "a" * 64,
) -> CalibratedTask:
    observations = [
        QueryObservation(
            run=index,
            execution_time_ms=value,
            planning_time_ms=0.1,
            plan_sha256=plan,
            shared_hit_blocks=10,
        )
        for index, value in enumerate(times, start=1)
    ]
    return CalibratedTask(
        task_id=task_id,
        template_id="01",
        completed_at_utc="2026-09-08T00:00:00Z",
        statement_timeout_ms=300_000,
        worker=worker,
        # A deliberately extreme warmup must have no effect on the analysis.
        warmups=[observations[0].model_copy(update={"execution_time_ms": 1_000_000.0})],
        measurements=observations,
        summary=CalibrationSummary(
            measurement_count=len(times),
            warmup_stable=True,
            median_execution_time_ms=statistics.median(times),
            mean_execution_time_ms=statistics.mean(times),
            sample_standard_deviation_ms=statistics.stdev(times),
            coefficient_of_variation=statistics.stdev(times) / statistics.mean(times),
            minimum_execution_time_ms=min(times),
            maximum_execution_time_ms=max(times),
            distinct_plan_count=1,
            plan_sha256s=[plan],
        ),
        representative_explain_analyze={"Plan": {"Node Type": "Seq Scan"}},
    )


def write_calibration(
    directory: Path, tasks: list[CalibrationTaskResult], *, num_trials: int = 6
) -> Path:
    (directory / "tasks").mkdir(parents=True)
    failed = sum(isinstance(task, FailedCalibrationTask) for task in tasks)
    manifest = CalibrationReport(
        benchmark_id=BenchmarkId.JOB,
        status=RunStatus.COMPLETED_WITH_FAILURES if failed else RunStatus.COMPLETED,
        started_at_utc="2026-09-08T00:00:00Z",
        completed_at_utc="2026-09-08T00:01:00Z",
        measurement=CalibrationSettings(
            max_warmup_runs=5,
            num_trials=num_trials,
            default_timeout_seconds=300.0,
        ),
        statement_timeout_ms=300_000,
        minimum_warmup_runs=2,
        buffer_stability_relative_tolerance=0.02,
        plan_fingerprint_version=4,
        task_count=len(tasks),
        completed_task_count=len(tasks) - failed,
        failed_task_count=failed,
    )
    (directory / "calibration.json").write_text(manifest.model_dump_json())
    for task in tasks:
        (directory / "tasks" / f"{task.task_id}.json").write_text(
            task.model_dump_json()
        )
    return directory


def test_constant_trials_remain_distinct_observations() -> None:
    result = estimate_noop_error([100.0] * 20, AnalysisSettings())
    assert result.window_count == 15
    assert result.comparison_count == 120
    assert result.estimated_noop_error_rate == 0
    assert result.absolute_log_ratio_p95 == 0


def test_all_assignments_cross_threshold_for_opposite_values_in_each_pair() -> None:
    result = estimate_noop_error([1.0, 2.0] * 3, AnalysisSettings())
    # Each side's median is the opposite of the other's, for all eight assignments.
    assert result.comparison_count == 8
    assert result.estimated_noop_error_rate == 1
    assert result.absolute_log_ratio_p95 == pytest.approx(math.log(2))


def test_pair_shared_drift_cancels_and_trial_order_matters() -> None:
    settings = AnalysisSettings()
    drifting = estimate_noop_error([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], settings)
    reordered = estimate_noop_error([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], settings)
    assert drifting.estimated_noop_error_rate == 0
    assert reordered.estimated_noop_error_rate > 0


def test_threshold_is_strict_and_even_pair_counts_use_arithmetic_medians() -> None:
    assert (
        estimate_noop_error(
            [1.0, 2.0] * 3, AnalysisSettings(tau=math.log(2))
        ).estimated_noop_error_rate
        == 0
    )
    # Of four assignments, two yield 1 versus 3; two yield 2 versus 2.
    result = estimate_noop_error(
        [1.0, 3.0] * 2, AnalysisSettings(paired_measurements=2)
    )
    assert result.estimated_noop_error_rate == 0.5


def test_percentile_interpolation() -> None:
    assert percentile([1.0], 0.95) == 1
    assert percentile([0.0, 1.0], 0.95) == pytest.approx(0.95)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_estimator_rejects_unusable_times(value: float) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        estimate_noop_error([1.0] * 5 + [value], AnalysisSettings())


def test_exact_estimator_fails_clearly_instead_of_hanging_or_switching_methods() -> (
    None
):
    with pytest.raises(ValueError, match="at least 6"):
        estimate_noop_error([1.0] * 5, AnalysisSettings())
    with pytest.raises(ValueError, match="exact enumeration exceeds"):
        estimate_noop_error([1.0] * 40, AnalysisSettings(paired_measurements=20))


def test_reader_excludes_failed_zero_and_changing_plan_tasks(
    tmp_path: Path, worker: WorkerManifest
) -> None:
    valid = task_record(worker, [100.0] * 6)
    changed = task_record(worker, [100.0] * 6, task_id="job-02a")
    changed = changed.model_copy(
        update={
            "measurements": [
                *changed.measurements[:-1],
                changed.measurements[-1].model_copy(update={"plan_sha256": "b" * 64}),
            ]
        }
    )
    zero = task_record(
        worker, [0.0, 100.0, 100.0, 100.0, 100.0, 100.0], task_id="job-03a"
    )
    failed = FailedCalibrationTask(
        task_id="job-04a",
        template_id="04",
        completed_at_utc=valid.completed_at_utc,
        statement_timeout_ms=valid.statement_timeout_ms,
        worker=worker,
        warmups=[],
        measurements=[],
        error_type="QueryTimeoutError",
        error="timed out",
    )
    directory = write_calibration(tmp_path / "source", [valid, changed, zero, failed])
    analysis = analyze_calibration(directory, AnalysisSettings())
    assert analysis.summary.eligible_task_count == 1
    assert analysis.summary.exclusions_by_reason == {
        "plan_changed": 1,
        "nonpositive_execution_time": 1,
        "failed_task": 1,
    }
    assert analysis.tasks[0].median_execution_time_ms == 100
    assert analysis.tasks[0].coefficient_of_variation == 0
    assert analysis.summary.mean_estimated_noop_error_rate == 0
    insufficient = analyze_calibration(
        directory, AnalysisSettings(paired_measurements=4)
    )
    assert insufficient.summary.eligible_task_count == 0
    assert insufficient.summary.mean_estimated_noop_error_rate is None
    assert insufficient.tasks[0].exclusion_reason == "insufficient_measurements"


@pytest.mark.parametrize(
    "corruption", ["missing", "counts", "order", "nan", "incomplete"]
)
def test_reader_rejects_incomplete_or_malformed_evidence(
    tmp_path: Path, worker: WorkerManifest, corruption: str
) -> None:
    directory = write_calibration(tmp_path / "source", [task_record(worker, [1.0] * 6)])
    path = directory / "tasks/job-01a.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "incomplete":
        path = directory / "calibration.json"
        data = json.loads(path.read_text())
        data["status"] = "running"
        path.write_text(json.dumps(data))
    else:
        data = json.loads(path.read_text())
        if corruption == "counts":
            data["summary"]["measurement_count"] = 5
        elif corruption == "order":
            data["measurements"].reverse()
        else:
            data["measurements"][0]["execution_time_ms"] = math.nan
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=str(directory)):
        analyze_calibration(directory, AnalysisSettings())


def test_comparison_matches_ids_and_separates_changed_plans(
    tmp_path: Path, worker: WorkerManifest
) -> None:
    primary = write_calibration(
        tmp_path / "a",
        [
            task_record(worker, [1.0, 2.0] * 3),
            task_record(worker, [1.0, 2.0] * 3, task_id="job-02a"),
            task_record(worker, [1.0] * 6, task_id="job-03a"),
        ],
    )
    comparison = write_calibration(
        tmp_path / "b",
        [
            task_record(worker, [1.0] * 6),
            task_record(worker, [1.0] * 6, task_id="job-02a", plan="b" * 64),
            task_record(worker, [1.0] * 6, task_id="job-04a"),
        ],
    )
    result = compare_calibrations(
        analyze_calibration(primary, AnalysisSettings()),
        analyze_calibration(comparison, AnalysisSettings()),
    )
    assert result.common_eligible_task_count == 2
    assert result.same_plan_task_count == 1
    assert result.mean_error_rate_delta_pp == -100
    assert result.same_plan_mean_error_rate_delta_pp == -100
    assert result.primary_only_task_ids == ["job-03a"]
    assert result.comparison_only_task_ids == ["job-04a"]
    assert [row.plan_changed for row in result.tasks] == [False, True]
    assert result.tasks[0].median_latency_ratio == pytest.approx(2 / 3)


def test_cli_writes_reproducible_reports_and_preserves_source(
    tmp_path: Path, worker: WorkerManifest, repository_root: Path
) -> None:
    directory = write_calibration(
        tmp_path / "source", [task_record(worker, [100.0] * 6)]
    )
    before = {path: path.read_bytes() for path in directory.rglob("*.json")}
    command = [
        sys.executable,
        "-m",
        "scripts.analysis.analyze_calibration",
        str(directory),
        "--compare",
        str(directory),
        "--paired-measurements",
        "3",
        "--tau",
        "0.05",
        "--output-dir",
        str(tmp_path / "report"),
    ]
    result = subprocess.run(
        command, cwd=repository_root, capture_output=True, text=True, check=True
    )
    saved = (tmp_path / "report/noise.json").read_bytes()
    report = AnalysisReport.model_validate_json(saved)
    assert report.primary.summary.eligible_task_count == 1
    assert report.settings.tau == 0.05
    assert report.paired_comparison is not None
    assert report.paired_comparison.mean_error_rate_delta_pp == 0
    assert "independent sample size" in (tmp_path / "report/noise.md").read_text()
    assert "Mean estimated no-op error rate: 0.00%" in result.stdout
    subprocess.run(
        command, cwd=repository_root, capture_output=True, text=True, check=True
    )
    assert (tmp_path / "report/noise.json").read_bytes() == saved
    assert {path: path.read_bytes() for path in directory.rglob("*.json")} == before


def test_cli_rejects_bad_settings_and_output_inside_task_records(
    tmp_path: Path, worker: WorkerManifest
) -> None:
    directory = write_calibration(tmp_path / "source", [task_record(worker, [1.0] * 6)])
    for extra in (["--tau", "nan"], ["--paired-measurements", "0"], []):
        with pytest.raises(SystemExit) as error:
            main([str(directory), "--output-dir", str(directory / "tasks"), *extra])
        assert error.value.code == 2
    assert not (directory / "tasks/noise.json").exists()
