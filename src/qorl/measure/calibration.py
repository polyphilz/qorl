from __future__ import annotations

import os
import platform
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.measure.query import (
    BUFFER_STABILITY_TOLERANCE,
    MIN_WARMUP_RUNS,
    measure_query,
)
from qorl.measure.run import TaskRun
from qorl.measure.schemas import QueryObservation, RunStatus
from qorl.measure.timeouts import GLOBAL_TIMEOUT_MS
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.taskset.schemas import Task
from qorl.taskset.taskset import TaskSet
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json
from qorl.util.time import utc_now
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import WorkerSlot

DEFAULT_NUM_TRIALS = 20
DEFAULT_MAX_WARMUP_RUNS = 5
MIN_CALIBRATION_TRIALS = 2


def validate_run_counts(max_warmup_runs: int, num_trials: int) -> None:
    """Require enough runs to compare warmups and compute sample variation."""
    if max_warmup_runs < MIN_WARMUP_RUNS:
        raise ValueError(f"max_warmup_runs must be at least {MIN_WARMUP_RUNS}")
    if num_trials < MIN_CALIBRATION_TRIALS:
        raise ValueError(f"num_trials must be at least {MIN_CALIBRATION_TRIALS}")


def calibrate_task(
    worker: PostgresClient,
    task_set: TaskSet,
    task: Task,
    *,
    max_warmup_runs: int,
    num_trials: int,
) -> dict[str, Any]:
    validate_run_counts(max_warmup_runs, num_trials)
    sql = task_set.load_sql(task)
    warmups: list[QueryObservation] = []
    measurements: list[QueryObservation] = []
    representative_explain: dict[str, Any] | None = None
    for result in measure_query(
        lambda: worker.explain(sql, GLOBAL_TIMEOUT_MS, analyze=True),
        max_warmup_runs=max_warmup_runs,
        num_trials=num_trials,
    ):
        if result.is_warmup:
            warmups.append(result.observation)
        else:
            if representative_explain is None:
                representative_explain = result.explain.document
            measurements.append(result.observation)

    execution_times = [item.execution_time_ms for item in measurements]
    mean = statistics.mean(execution_times)
    standard_deviation = statistics.stdev(execution_times)
    fingerprints = sorted({item.plan_sha256 for item in measurements})
    return {
        "schema_version": 1,
        "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
        "task_id": task.task_id,
        "template_id": task.template_id,
        "status": RunStatus.COMPLETED.value,
        "completed_at_utc": utc_now(),
        "warmups": [item.model_dump() for item in warmups],
        "measurements": [item.model_dump() for item in measurements],
        "summary": {
            "measurement_count": len(measurements),
            "median_execution_time_ms": statistics.median(execution_times),
            "mean_execution_time_ms": mean,
            "sample_standard_deviation_ms": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean,
            "minimum_execution_time_ms": min(execution_times),
            "maximum_execution_time_ms": max(execution_times),
            "distinct_plan_count": len(fingerprints),
            "plan_sha256s": fingerprints,
        },
        "representative_explain_analyze": representative_explain,
    }


def calibrate_on_worker(
    pool: ContainerPool,
    task_set: TaskSet,
    task: Task,
    *,
    max_warmup_runs: int,
    num_trials: int,
) -> tuple[WorkerSlot, dict[str, Any]]:
    with pool.claim_worker() as slot:
        result = calibrate_task(
            slot.client,
            task_set,
            task,
            max_warmup_runs=max_warmup_runs,
            num_trials=num_trials,
        )
        result["worker"] = slot.resources.manifest()
        return slot, result


def failed_task(task: Task, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "template_id": task.template_id,
        "status": RunStatus.FAILED.value,
        "completed_at_utc": utc_now(),
        "error": str(error),
    }


def calibrate(
    repository: Path,
    *,
    postgres_config_path: Path,
    pool_config_path: Path,
    max_warmup_runs: int = DEFAULT_MAX_WARMUP_RUNS,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> Path:
    """Calibrate the complete JOB benchmark and write per-task and pool reports."""
    validate_run_counts(max_warmup_runs, num_trials)
    postgres_config = PostgresConfig.load(postgres_config_path)
    pool_config = load_pool_config(repository, pool_config_path)
    task_set = TaskSet.load(repository, "job")
    tasks = task_set.tasks
    worker_count = len(pool_config.workers)

    started_at = datetime.now(UTC)
    calibration_id = started_at.strftime(
        f"{task_set.task_set_id}-{postgres_config.config_id}"
        f"-{pool_config.profile_id}-%Y%m%dT%H%M%SZ"
    )
    output_dir = repository / "outputs/calibration" / calibration_id
    output_dir.mkdir(parents=True, exist_ok=False)
    task_dir = output_dir / "tasks"

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "calibration_id": calibration_id,
        "purpose": "evaluation baseline calibration",
        "status": RunStatus.RUNNING.value,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "inventory_id": task_set.task_set_id,
        "task_set_id": task_set.task_set_id,
        "inventory_sha256": sha256_file(task_set.inventory_path),
        "data_identity": {"fixture_id": task_set.fixture_id},
        "runtime_identity": {"postgres_config_id": postgres_config.config_id},
        "postgres_config": postgres_config.manifest().model_dump(),
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
        },
        "protocol": {
            "explain": "EXPLAIN (ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON)",
            "statement_timeout_ms": GLOBAL_TIMEOUT_MS,
            "minimum_warmup_runs": MIN_WARMUP_RUNS,
            "maximum_warmup_runs": max_warmup_runs,
            "buffer_stability_relative_tolerance": BUFFER_STABILITY_TOLERANCE,
            "measurement_runs": num_trials,
            "coefficient_of_variation": "sample standard deviation / arithmetic mean",
            "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
            "worker_count": worker_count,
            "concurrent_tasks": min(worker_count, len(tasks)),
            "one_query_per_worker": True,
        },
        "selection": None,
        "worker_pool": None,
        "task_count": len(tasks),
        "completed_task_count": 0,
        "failed_task_count": 0,
    }
    manifest_path = output_dir / "calibration.json"
    write_json(manifest_path, manifest)

    project_name = f"qorl-cal-{started_at:%Y%m%d%H%M%S}-{os.getpid()}".lower()
    failures = 0
    run = TaskRun(
        repository,
        project_name,
        output_dir,
        manifest_path,
        manifest,
        pool_field="worker_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
    )

    def execute_task(
        pool: ContainerPool, task: Task
    ) -> tuple[WorkerSlot, dict[str, Any]]:
        return calibrate_on_worker(
            pool,
            task_set,
            task,
            max_warmup_runs=max_warmup_runs,
            num_trials=num_trials,
        )

    try:
        with run:
            for completion in run.map(
                tasks,
                execute_task,
                handled_errors=(PostgresError, ContainerError),
            ):
                task = completion.item
                task_id = task.task_id
                print(
                    f"[{completion.ordinal}/{manifest['task_count']}] {task_id}",
                    flush=True,
                )
                if completion.error is None:
                    if completion.result is None:
                        raise RuntimeError("calibration task returned no result")
                    slot, result = completion.result
                    summary = result["summary"]
                    print(
                        f"  worker={slot.resources.index} "
                        f"median={summary['median_execution_time_ms']:.3f} ms "
                        f"cv={summary['coefficient_of_variation']:.4f}"
                    )
                    manifest["completed_task_count"] += 1
                else:
                    if not isinstance(
                        completion.error, (PostgresError, ContainerError)
                    ):
                        raise completion.error
                    failures += 1
                    result = failed_task(task, completion.error)
                    manifest["failed_task_count"] += 1
                    print(f"  failed: {completion.error}")
                write_json(task_dir / f"{task_id}.json", result)
                run.write()
    except BaseException:
        manifest["status"] = RunStatus.INTERRUPTED.value
        manifest["completed_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = (
        RunStatus.COMPLETED.value
        if failures == 0
        else RunStatus.COMPLETED_WITH_FAILURES.value
    )
    manifest["completed_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(
            f"calibration completed with {failures} failed tasks; results: {output_dir}"
        )
    return output_dir
