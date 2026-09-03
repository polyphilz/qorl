from __future__ import annotations

import json
import math
import os
import platform
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot, start_pool
from qorl.db.worker import PostgresWorker, WorkerError
from qorl.evaluation.baselines.random import sample_action, sampler_manifest
from qorl.evaluation.types import PolicyType, RunStatus
from qorl.measure.rollout import (
    DEFAULT_MEASUREMENTS,
    FINAL_PAIRS,
    GLOBAL_TIMEOUT_MS,
    MAX_CANDIDATES,
    RIGOROUS_EVALUATION_PROTOCOL_V1,
    RolloutEvaluator,
)
from qorl.measure.schemas import FinalStatus
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.util.hashing import sha256_file
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet

DEFAULT_RUN_CONFIG = "experiments/000-vanilla-baseline/run.json"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        result
        for result in results
        if result.get("final", {}).get("status") == FinalStatus.COMPLETED
    ]
    scores = [result["final"]["score"] for result in completed]
    candidate_time = sum(
        result["final"]["candidate_median_execution_time_ms"] for result in completed
    )
    default_time = sum(
        result["final"]["default_median_execution_time_ms"] for result in completed
    )
    attempts = [
        candidate for result in results for candidate in result.get("candidates", [])
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
    task_set: TaskSet,
    task: dict[str, Any],
    policy: dict[str, Any],
    agent: QoAgentPolicy | None,
) -> dict[str, Any]:
    evaluator = RolloutEvaluator(worker, task_set, task)
    baseline = evaluator.start()
    trace: dict[str, Any]
    if policy["type"] == PolicyType.RANDOM_STRUCTURED_ACTION:
        action_rng = random.Random(f"{policy['seed']}:{task['task_id']}:actions")
        for _ in range(MAX_CANDIDATES):
            candidate = evaluator.evaluate(sample_action(evaluator.catalog, action_rng))
            if candidate.constraints_satisfied:
                print(
                    f"  {candidate.candidate_id}: {candidate.provisional_speedup:.3f}x"
                )
            else:
                print(f"  {candidate.candidate_id}: invalid")
        trace = {"random_seed": policy["seed"]}
    else:
        if agent is None:
            raise RuntimeError("qo-agent policy is not initialized")
        trace = agent.search(evaluator)

    final = evaluator.finish(random.Random(f"{policy['seed']}:{task['task_id']}:pairs"))
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "status": final.status,
        "completed_at_utc": utc_now(),
        "policy_trace": trace,
        "default": baseline.to_wire(),
        "candidates": [candidate.to_wire() for candidate in evaluator.candidates],
        "final": final.to_wire(),
    }


def run_task_on_worker(
    pool: WorkerPool,
    task_set: TaskSet,
    task: dict[str, Any],
    policy: dict[str, Any],
    agent: QoAgentPolicy | None,
) -> tuple[WorkerSlot, dict[str, Any]]:
    with pool.claim_worker() as slot:
        result = run_task(slot.worker, task_set, task, policy, agent)
        result["worker"] = slot.resources.manifest()
        return slot, result


def load_run_config(
    repository: Path, configured: str | None = None
) -> tuple[Path, dict[str, Any]]:
    configured = Path(
        configured or os.environ.get("QORL_RUN_CONFIG", DEFAULT_RUN_CONFIG)
    )
    path = configured if configured.is_absolute() else repository / configured
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load run configuration {path}: {error}") from error
    if value.get("schema_version") != 1:
        raise RuntimeError("run configuration schema_version must equal 1")
    policy_configured = value.get("policy_config")
    if not isinstance(policy_configured, str) or not policy_configured:
        raise RuntimeError("run configuration must name a policy_config")
    policy_path = Path(policy_configured)
    if not policy_path.is_absolute():
        policy_path = repository / policy_path
    try:
        policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot load policy configuration {policy_path}: {error}"
        ) from error
    if policy_value.get("schema_version") != 1:
        raise RuntimeError("policy configuration schema_version must equal 1")
    policy = policy_value.get("policy")
    if not isinstance(policy, dict) or policy.get("type") not in {
        PolicyType.RANDOM_STRUCTURED_ACTION,
        PolicyType.QO_AGENT,
    }:
        raise RuntimeError("policy configuration has an unknown policy type")
    seed = policy.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("policy configuration must define an integer seed")
    run_id_prefix = value.get("run_id_prefix")
    if not isinstance(run_id_prefix, str) or not run_id_prefix:
        raise RuntimeError("run configuration must define run_id_prefix")
    return path, {
        **value,
        "policy": policy,
        "_policy_config_path": policy_path,
    }


def run_benchmark(repository: Path, configured: str | None = None) -> Path:
    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "job-v1", fixture.data_identity)
    config_path, config = load_run_config(repository, configured)
    policy = config["policy"]
    agent: QoAgentPolicy | None = None
    if policy["type"] == PolicyType.QO_AGENT:
        agent = QoAgentPolicy(QoAgentConfig.from_dict(policy))
        agent.preflight()
    started_at = datetime.now(UTC)
    benchmark_id = started_at.strftime(f"{config['run_id_prefix']}-%Y%m%dT%H%M%SZ")
    output_dir = repository / "outputs/runs" / benchmark_id
    output_dir.mkdir(parents=True, exist_ok=False)
    task_dir = output_dir / "tasks"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "status": RunStatus.RUNNING.value,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "inventory_id": task_set.inventory["inventory_id"],
        "inventory_sha256": sha256_file(task_set.inventory_path),
        "data_identity": fixture.data_identity,
        "runtime_identity": fixture.runtime_identity,
        "snapshot_manifest_sha256": sha256_file(fixture.snapshot_manifest_path),
        "run_config": {
            "path": str(config_path.relative_to(repository)),
            "sha256": sha256_file(config_path),
        },
        "policy_config": {
            "path": str(config["_policy_config_path"].relative_to(repository)),
            "sha256": sha256_file(config["_policy_config_path"]),
        },
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
        },
        "policy": (
            {
                **policy,
                "candidate_count": MAX_CANDIDATES,
                "sampler": sampler_manifest(),
            }
            if policy["type"] == PolicyType.RANDOM_STRUCTURED_ACTION
            else {**agent.manifest(), "candidate_count": MAX_CANDIDATES}
        ),
        "protocol": {
            "id": RIGOROUS_EVALUATION_PROTOCOL_V1.protocol_id,
            "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
            "default_warmup_runs": 1,
            "default_measurement_runs": DEFAULT_MEASUREMENTS,
            "novel_candidate_warmup_runs": 1,
            "novel_candidate_measurement_runs": 1,
            "final_warmup_runs_per_plan": 1,
            "final_randomized_pair_count": FINAL_PAIRS,
            "max_explain_analyze_executions": (
                RIGOROUS_EVALUATION_PROTOCOL_V1.max_explain_analyze_executions
            ),
            "global_timeout_ms": GLOBAL_TIMEOUT_MS,
            "task_timeout": (
                "min(global_timeout_ms, max(5000, 3 * provisional_default_median_ms))"
            ),
            "score": "clip(default_median / candidate_median, 0.1, 10)",
        },
        "worker_pool": None,
        "task_count": task_set.inventory["task_count"],
        "completed_task_count": 0,
        "failed_task_count": 0,
        "summary": None,
    }
    manifest_path = output_dir / "run.json"
    write_json(manifest_path, manifest)

    project_name = f"qorl-run-{started_at:%Y%m%d%H%M%S}-{os.getpid()}".lower()
    tasks = task_set.inventory["tasks"]
    results_by_task: dict[str, dict[str, Any]] = {}
    pool: WorkerPool | None = None
    try:
        pool = start_pool(fixture, project_name)
        manifest["worker_pool"] = pool.manifest()
        manifest["protocol"].update(
            {
                "worker_count": len(pool.workers),
                "concurrent_tasks": min(len(pool.workers), len(tasks)),
                "one_query_per_worker": True,
            }
        )
        write_json(manifest_path, manifest)
        for slot in pool.workers:
            slot.worker.capture_environment(
                output_dir / f"worker-{slot.resources.index}", "pre"
            )

        with ThreadPoolExecutor(max_workers=len(pool.workers)) as executor:
            futures: dict[Future[tuple[WorkerSlot, dict[str, Any]]], dict[str, Any]] = {
                executor.submit(
                    run_task_on_worker, pool, task_set, task, policy, agent
                ): task
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                task_id = task["task_id"]
                print(f"[{index}/{manifest['task_count']}] {task_id}", flush=True)
                try:
                    slot, result = future.result()
                    if result["status"] == RunStatus.COMPLETED:
                        manifest["completed_task_count"] += 1
                        print(
                            f"  worker={slot.resources.index} "
                            f"final={result['final']['score']:.3f}x"
                        )
                    else:
                        manifest["failed_task_count"] += 1
                        print("  no valid candidate")
                except (ModelError, WorkerError) as error:
                    result = {
                        "schema_version": 1,
                        "task_id": task["task_id"],
                        "template_id": task["template_id"],
                        "status": RunStatus.FAILED.value,
                        "completed_at_utc": utc_now(),
                        "error": str(error),
                    }
                    manifest["failed_task_count"] += 1
                    print(f"  failed: {error}")
                results_by_task[task_id] = result
                write_json(task_dir / f"{task_id}.json", result)
                write_json(manifest_path, manifest)

        for slot in pool.workers:
            slot.worker.capture_environment(
                output_dir / f"worker-{slot.resources.index}", "post"
            )
        pool.close()
        pool = None
    except BaseException:
        if pool is not None:
            pool.close()
        manifest["status"] = RunStatus.INTERRUPTED.value
        manifest["completed_at_utc"] = utc_now()
        manifest["summary"] = summarize(
            [
                results_by_task[task["task_id"]]
                for task in tasks
                if task["task_id"] in results_by_task
            ]
        )
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = (
        RunStatus.COMPLETED.value
        if manifest["failed_task_count"] == 0
        else RunStatus.COMPLETED_WITH_FAILURES.value
    )
    manifest["completed_at_utc"] = utc_now()
    manifest["summary"] = summarize(
        [results_by_task[task["task_id"]] for task in tasks]
    )
    write_json(manifest_path, manifest)
    return output_dir
