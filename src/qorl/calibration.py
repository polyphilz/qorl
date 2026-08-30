from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.fixture import DatabaseFixture, TaskSet, sha256_file
from qorl.worker import PostgresWorker, WorkerError


MEASUREMENT_RUNS = 20
MIN_WARMUP_RUNS = 2
MAX_WARMUP_RUNS = 5
BUFFER_STABILITY_TOLERANCE = 0.02
STATEMENT_TIMEOUT_MS = 120_000
PLAN_FINGERPRINT_VERSION = 2

RUNTIME_PLAN_KEYS = {
    "Cache Evictions",
    "Cache Hits",
    "Cache Misses",
    "Cache Overflows",
    "Hash Batches",
    "Hash Buckets",
    "Heap Fetches",
    "I/O Read Time",
    "I/O Write Time",
    "Index Searches",
    "Maximum Storage",
    "Original Hash Batches",
    "Original Hash Buckets",
    "Peak Memory Usage",
    "Sort Method",
    "Sort Space Type",
    "Sort Space Used",
    "Storage",
    "Workers",
    "Workers Launched",
}
RUNTIME_PLAN_PREFIXES = ("Actual ", "Rows Removed by ", "WAL ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def canonical_plan(value: Any) -> Any:
    if isinstance(value, list):
        return [canonical_plan(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: canonical_plan(item)
        for key, item in value.items()
        if key not in RUNTIME_PLAN_KEYS
        and not key.startswith(RUNTIME_PLAN_PREFIXES)
        and not key.endswith(" Blocks")
    }


def plan_sha256(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_plan(plan), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observation(explain: dict[str, Any], run_number: int) -> dict[str, Any]:
    plan = explain["Plan"]
    return {
        "run": run_number,
        "execution_time_ms": explain["Execution Time"],
        "planning_time_ms": explain["Planning Time"],
        "shared_hit_blocks": plan.get("Shared Hit Blocks", 0),
        "shared_read_blocks": plan.get("Shared Read Blocks", 0),
        "plan_sha256": plan_sha256(plan),
    }


def buffers_stable(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous["plan_sha256"] != current["plan_sha256"]:
        return False
    for key in ("shared_hit_blocks", "shared_read_blocks"):
        left = previous[key]
        right = current[key]
        scale = max(1, left, right)
        if abs(left - right) / scale > BUFFER_STABILITY_TOLERANCE:
            return False
    return True


def calibrate_task(
    worker: PostgresWorker, task_set: TaskSet, task: dict[str, Any]
) -> dict[str, Any]:
    sql = task_set.load_sql(task)
    warmups: list[dict[str, Any]] = []
    for run_number in range(1, MAX_WARMUP_RUNS + 1):
        result = observation(
            worker.explain_analyze(sql, STATEMENT_TIMEOUT_MS), run_number
        )
        warmups.append(result)
        if (
            run_number >= MIN_WARMUP_RUNS
            and buffers_stable(warmups[-2], warmups[-1])
        ):
            break

    measurements: list[dict[str, Any]] = []
    representative_explain: dict[str, Any] | None = None
    for run_number in range(1, MEASUREMENT_RUNS + 1):
        explain = worker.explain_analyze(sql, STATEMENT_TIMEOUT_MS)
        if representative_explain is None:
            representative_explain = explain
        measurements.append(observation(explain, run_number))

    execution_times = [item["execution_time_ms"] for item in measurements]
    mean = statistics.mean(execution_times)
    standard_deviation = statistics.stdev(execution_times)
    fingerprints = sorted({item["plan_sha256"] for item in measurements})
    return {
        "schema_version": 1,
        "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "status": "completed",
        "completed_at_utc": utc_now(),
        "warmups": warmups,
        "measurements": measurements,
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


def calibrate(repository: Path) -> Path:
    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "job-v1", fixture.identity)
    started_at = datetime.now(timezone.utc)
    calibration_id = started_at.strftime("job-v1-%Y%m%dT%H%M%SZ")
    output_dir = repository / "outputs/calibration" / calibration_id
    output_dir.mkdir(parents=True, exist_ok=False)
    task_dir = output_dir / "tasks"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "inventory_id": task_set.inventory["inventory_id"],
        "inventory_sha256": sha256_file(task_set.inventory_path),
        "database": task_set.inventory["database"],
        "snapshot_manifest_sha256": sha256_file(fixture.snapshot_manifest_path),
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
        },
        "protocol": {
            "explain": "EXPLAIN (ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON)",
            "statement_timeout_ms": STATEMENT_TIMEOUT_MS,
            "minimum_warmup_runs": MIN_WARMUP_RUNS,
            "maximum_warmup_runs": MAX_WARMUP_RUNS,
            "buffer_stability_relative_tolerance": BUFFER_STABILITY_TOLERANCE,
            "measurement_runs": MEASUREMENT_RUNS,
            "coefficient_of_variation": "sample standard deviation / arithmetic mean",
            "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
        },
        "task_count": task_set.inventory["task_count"],
        "completed_task_count": 0,
        "failed_task_count": 0,
    }
    manifest_path = output_dir / "calibration.json"
    write_json(manifest_path, manifest)

    project_name = f"qorl-cal-{started_at:%Y%m%d%H%M%S}-{os.getpid()}".lower()
    failures = 0
    try:
        with PostgresWorker(fixture, project_name) as worker:
            worker.capture_environment(output_dir, "pre")
            for index, task in enumerate(task_set.inventory["tasks"], start=1):
                task_id = task["task_id"]
                print(f"[{index}/{manifest['task_count']}] {task_id}", flush=True)
                try:
                    result = calibrate_task(worker, task_set, task)
                    summary = result["summary"]
                    print(
                        f"  median={summary['median_execution_time_ms']:.3f} ms "
                        f"cv={summary['coefficient_of_variation']:.4f}"
                    )
                    manifest["completed_task_count"] += 1
                except WorkerError as error:
                    failures += 1
                    result = {
                        "schema_version": 1,
                        "task_id": task_id,
                        "template_id": task["template_id"],
                        "status": "failed",
                        "completed_at_utc": utc_now(),
                        "error": str(error),
                    }
                    manifest["failed_task_count"] += 1
                    print(f"  failed: {error}")
                write_json(task_dir / f"{task_id}.json", result)
                write_json(manifest_path, manifest)
            worker.capture_environment(output_dir, "post")
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["completed_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "completed" if failures == 0 else "completed_with_failures"
    manifest["completed_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(
            f"calibration completed with {failures} failed tasks; results: {output_dir}"
        )
    return output_dir
