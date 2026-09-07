"""Characterize selected default queries using a configured PostgreSQL pool."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from functools import partial
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev
from threading import Event
from uuid import uuid4

from qorl.measure.environment import capture_environment
from qorl.measure.query import (
    BUFFER_STABILITY_TOLERANCE,
    MIN_WARMUP_RUNS,
    buffers_stable,
    measure_query,
)
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
from qorl.paths import REPOSITORY_ROOT
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import ExplainResult
from qorl.taskset.schemas import Task, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.io import write_json
from qorl.util.time import utc_now
from qorl.worker_pool.containers import ContainerPool, start_pool
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import PoolConfig

MILLISECONDS_PER_SECOND = 1_000


def calibrate_task(
    pool: ContainerPool,
    task_set: TaskSet,
    task: Task,
    settings: CalibrationSettings,
    stop: Event,
) -> CalibrationTaskResult:
    """Hold one worker for a task; preserve completed observations if SQL fails."""
    timeout_ms = ceil(settings.default_timeout_seconds * MILLISECONDS_PER_SECOND)
    warmups: list[QueryObservation] = []
    measurements: list[QueryObservation] = []
    representative: ExplainResult | None = None
    with pool.claim_worker() as slot:
        sql = task_set.load_sql(task)

        def execute() -> ExplainResult:
            if stop.is_set():
                raise CancelledError("calibration interrupted")
            return slot.client.explain(sql, timeout_ms, analyze=True)

        try:
            for result in measure_query(
                execute,
                max_warmup_runs=settings.max_warmup_runs,
                num_trials=settings.num_trials,
            ):
                if result.is_warmup:
                    warmups.append(result.observation)
                else:
                    if representative is None:
                        representative = result.explain
                    measurements.append(result.observation)
        except (PostgresError, ContainerError) as error:
            return FailedCalibrationTask(
                task_id=task.task_id,
                template_id=task.template_id,
                completed_at_utc=utc_now(),
                statement_timeout_ms=timeout_ms,
                worker=slot.resources.manifest(),
                warmups=warmups,
                measurements=measurements,
                error_type=type(error).__name__,
                error=str(error),
            )

        if representative is None:
            raise RuntimeError("calibration produced no measured trials")
        execution_times = [item.execution_time_ms for item in measurements]
        average = mean(execution_times)
        deviation = stdev(execution_times)
        fingerprints = sorted({item.plan_sha256 for item in measurements})
        return CalibratedTask(
            task_id=task.task_id,
            template_id=task.template_id,
            completed_at_utc=utc_now(),
            statement_timeout_ms=timeout_ms,
            worker=slot.resources.manifest(),
            warmups=warmups,
            measurements=measurements,
            summary=CalibrationSummary(
                measurement_count=len(measurements),
                warmup_stable=buffers_stable(warmups[-2], warmups[-1]),
                median_execution_time_ms=median(execution_times),
                mean_execution_time_ms=average,
                sample_standard_deviation_ms=deviation,
                coefficient_of_variation=deviation / average if average else None,
                minimum_execution_time_ms=min(execution_times),
                maximum_execution_time_ms=max(execution_times),
                distinct_plan_count=len(fingerprints),
                plan_sha256s=fingerprints,
            ),
            representative_explain_analyze=representative.document,
        )


def calibrate(
    task_set: TaskSet,
    selection: TaskSelection,
    output_dir: Path,
    *,
    settings: CalibrationSettings,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> CalibrationReport:
    """Write a new calibration stage, continuing after individual query failures."""
    tasks = task_set.resolve(selection)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "calibration.json"
    report = CalibrationReport(
        benchmark_id=selection.benchmark_id,
        status=RunStatus.RUNNING,
        started_at_utc=utc_now(),
        measurement=settings,
        statement_timeout_ms=ceil(
            settings.default_timeout_seconds * MILLISECONDS_PER_SECOND
        ),
        minimum_warmup_runs=MIN_WARMUP_RUNS,
        buffer_stability_relative_tolerance=BUFFER_STABILITY_TOLERANCE,
        plan_fingerprint_version=PLAN_FINGERPRINT_VERSION,
        task_count=len(tasks),
    )
    write_json(report_path, report.model_dump(mode="json"))
    pool: ContainerPool | None = None
    stop = Event()
    try:
        pool = start_pool(
            f"qorl-cal-{uuid4().hex}",
            REPOSITORY_ROOT / "data/imdb.tar.gz",
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
        report.worker_pool = pool.manifest()
        write_json(report_path, report.model_dump(mode="json"))
        for slot in pool.workers:
            capture_environment(
                pool, slot, output_dir / f"worker-{slot.resources.index}", "pre"
            )
        with ThreadPoolExecutor(max_workers=len(pool.workers)) as executor:
            measure = partial(
                calibrate_task, pool, task_set, settings=settings, stop=stop
            )
            futures: list[Future[CalibrationTaskResult]] = []
            try:
                for task in tasks:
                    futures.append(executor.submit(measure, task))
                for ordinal, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    write_json(
                        output_dir / "tasks" / f"{result.task_id}.json",
                        result.model_dump(mode="json"),
                    )
                    if isinstance(result, CalibratedTask):
                        report.completed_task_count += 1
                        detail = (
                            f"median={result.summary.median_execution_time_ms:.3f} ms "
                            f"cv={result.summary.coefficient_of_variation}"
                        )
                    else:
                        report.failed_task_count += 1
                        detail = f"failed: {result.error}"
                    print(
                        f"[{ordinal}/{len(tasks)}] {result.task_id} "
                        f"worker={result.worker.slot} {detail}",
                        flush=True,
                    )
                    write_json(report_path, report.model_dump(mode="json"))
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        for slot in pool.workers:
            capture_environment(
                pool, slot, output_dir / f"worker-{slot.resources.index}", "post"
            )
    except BaseException as error:
        report.status = (
            RunStatus.INTERRUPTED
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else RunStatus.FAILED
        )
        report.error = str(error) or type(error).__name__
        report.completed_at_utc = utc_now()
        write_json(report_path, report.model_dump(mode="json"))
        raise
    finally:
        if pool is not None:
            pool.close()

    report.status = (
        RunStatus.COMPLETED_WITH_FAILURES
        if report.failed_task_count
        else RunStatus.COMPLETED
    )
    report.completed_at_utc = utc_now()
    write_json(report_path, report.model_dump(mode="json"))
    if report.failed_task_count:
        raise RuntimeError(
            f"calibration completed with {report.failed_task_count} failed tasks; "
            f"results: {output_dir}"
        )
    return report
