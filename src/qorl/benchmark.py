from __future__ import annotations

import math
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.calibration import PLAN_FINGERPRINT_VERSION, utc_now, write_json
from qorl.fixture import JobFixture, sha256_file
from qorl.random_policy import sample_action, sampler_manifest
from qorl.rollout import (
    DEFAULT_MEASUREMENTS,
    FINAL_PAIRS,
    GLOBAL_TIMEOUT_MS,
    MAX_CANDIDATES,
    RolloutEvaluator,
)
from qorl.worker import PostgresWorker, WorkerError


RUN_SEED = 20260827


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        result
        for result in results
        if result.get("final", {}).get("status") == "completed"
    ]
    scores = [result["final"]["score"] for result in completed]
    candidate_time = sum(
        result["final"]["candidate_median_execution_time_ms"]
        for result in completed
    )
    default_time = sum(
        result["final"]["default_median_execution_time_ms"]
        for result in completed
    )
    attempts = [
        candidate
        for result in results
        for candidate in result.get("candidates", [])
    ]
    failures = len(results) - len(completed)
    regressions = sum(value < 1.0 for value in scores)
    return {
        "task_count": len(results),
        "scored_task_count": len(completed),
        "failure_count": failures,
        "failure_rate": failures / len(results) if results else 0.0,
        "geometric_mean_speedup": (
            math.exp(sum(math.log(value) for value in scores) / len(scores))
            if scores
            else None
        ),
        "candidate_workload_time_ms": candidate_time,
        "default_workload_time_ms": default_time,
        "total_workload_speedup": (
            default_time / candidate_time if candidate_time else None
        ),
        "regression_count": regressions,
        "regression_rate": regressions / len(scores) if scores else None,
        "worst_regression_speedup": min(scores) if scores else None,
        "attempt_count": len(attempts),
        "invalid_attempt_count": sum(
            not candidate["constraints_satisfied"] for candidate in attempts
        ),
        "duplicate_attempt_count": sum(
            candidate.get("duplicate_of") is not None for candidate in attempts
        ),
    }


def run_task(
    worker: PostgresWorker,
    fixture: JobFixture,
    task: dict[str, Any],
) -> dict[str, Any]:
    evaluator = RolloutEvaluator(worker, fixture, task)
    baseline = evaluator.start()
    action_rng = random.Random(f"{RUN_SEED}:{task['task_id']}:actions")
    candidates: list[dict[str, Any]] = []
    for _ in range(MAX_CANDIDATES):
        candidate = evaluator.evaluate(
            sample_action(evaluator.catalog, action_rng)
        )
        candidates.append(candidate)
        if candidate["constraints_satisfied"]:
            print(
                f"  {candidate['candidate_id']}: "
                f"{candidate['provisional_speedup']:.3f}x"
            )
        else:
            print(f"  {candidate['candidate_id']}: invalid")

    final = evaluator.finish(
        random.Random(f"{RUN_SEED}:{task['task_id']}:pairs")
    )
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "status": final["status"],
        "completed_at_utc": utc_now(),
        "random_seed": RUN_SEED,
        "default": baseline,
        "candidates": candidates,
        "final": final,
    }


def run_random_benchmark(repository: Path) -> Path:
    fixture = JobFixture.load(repository)
    started_at = datetime.now(timezone.utc)
    benchmark_id = started_at.strftime("random-v1-%Y%m%dT%H%M%SZ")
    output_dir = repository / "outputs/runs" / benchmark_id
    output_dir.mkdir(parents=True, exist_ok=False)
    task_dir = output_dir / "tasks"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "inventory_id": fixture.inventory["inventory_id"],
        "inventory_sha256": sha256_file(fixture.inventory_path),
        "database": fixture.inventory["database"],
        "snapshot_manifest_sha256": sha256_file(
            fixture.snapshot_manifest_path
        ),
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
        },
        "policy": {
            "type": "random_structured_action",
            "seed": RUN_SEED,
            "candidate_count": MAX_CANDIDATES,
            "sampler": sampler_manifest(),
        },
        "protocol": {
            "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
            "default_warmup_runs": 1,
            "default_measurement_runs": DEFAULT_MEASUREMENTS,
            "novel_candidate_warmup_runs": 1,
            "novel_candidate_measurement_runs": 1,
            "final_warmup_runs_per_plan": 1,
            "final_randomized_pair_count": FINAL_PAIRS,
            "global_timeout_ms": GLOBAL_TIMEOUT_MS,
            "task_timeout": (
                "min(global_timeout_ms, max(5000, "
                "3 * provisional_default_median_ms))"
            ),
            "score": "clip(default_median / candidate_median, 0.1, 10)",
        },
        "task_count": fixture.inventory["task_count"],
        "completed_task_count": 0,
        "failed_task_count": 0,
        "summary": None,
    }
    manifest_path = output_dir / "run.json"
    write_json(manifest_path, manifest)

    project_name = (
        f"qorl-run-{started_at:%Y%m%d%H%M%S}-{os.getpid()}".lower()
    )
    results: list[dict[str, Any]] = []
    try:
        with PostgresWorker(fixture, project_name) as worker:
            worker.capture_environment(output_dir, "pre")
            for index, task in enumerate(
                fixture.inventory["tasks"], start=1
            ):
                print(
                    f"[{index}/{manifest['task_count']}] {task['task_id']}",
                    flush=True,
                )
                try:
                    result = run_task(worker, fixture, task)
                    if result["status"] == "completed":
                        manifest["completed_task_count"] += 1
                        print(f"  final={result['final']['score']:.3f}x")
                    else:
                        manifest["failed_task_count"] += 1
                        print("  no valid candidate")
                except WorkerError as error:
                    result = {
                        "schema_version": 1,
                        "task_id": task["task_id"],
                        "template_id": task["template_id"],
                        "status": "failed",
                        "completed_at_utc": utc_now(),
                        "error": str(error),
                    }
                    manifest["failed_task_count"] += 1
                    print(f"  failed: {error}")
                results.append(result)
                write_json(task_dir / f"{task['task_id']}.json", result)
                write_json(manifest_path, manifest)
            worker.capture_environment(output_dir, "post")
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["completed_at_utc"] = utc_now()
        manifest["summary"] = summarize(results)
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = (
        "completed"
        if manifest["failed_task_count"] == 0
        else "completed_with_failures"
    )
    manifest["completed_at_utc"] = utc_now()
    manifest["summary"] = summarize(results)
    write_json(manifest_path, manifest)
    return output_dir
