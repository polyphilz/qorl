from __future__ import annotations

import json
import math
import os
import platform
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.agent.types import PolicyType
from qorl.evaluation.baselines.random import sample_action, sampler_manifest
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.run import TaskRun
from qorl.measure.schemas import (
    DefaultDuplicateOutcome,
    KeptDefaultOutcome,
    MeasuredOutcome,
    OutcomeKind,
    RolloutMeasurementSettings,
    RolloutRecord,
    RunStatus,
)
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

MAX_CANDIDATES = 5

DEFAULT_RUN_CONFIG = "experiments/000-vanilla-baseline/run.json"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        RolloutRecord.model_validate_json(json.dumps(result["rollout"]))
        for result in results
        if "rollout" in result
    ]
    outcomes = [record.final for record in records if record.final is not None]
    speedups = [outcome.speedup for outcome in outcomes if outcome.speedup is not None]
    candidate_time = 0.0
    default_time = 0.0
    for record in records:
        final = record.final
        if isinstance(final, MeasuredOutcome):
            candidate_time += final.candidate_median_execution_time_ms
            default_time += final.default_median_execution_time_ms
        elif isinstance(final, (KeptDefaultOutcome, DefaultDuplicateOutcome)):
            if (
                record.default is not None
                and record.default.median_execution_time_ms is not None
            ):
                candidate_time += record.default.median_execution_time_ms
                default_time += record.default.median_execution_time_ms
    return {
        "task_count": len(results),
        "scored_task_count": len(speedups),
        "failure_count": len(results) - len(outcomes),
        "timeout_count": sum(
            outcome.kind == OutcomeKind.TIMED_OUT for outcome in outcomes
        ),
        "no_valid_candidate_count": sum(
            outcome.kind == OutcomeKind.NO_VALID_CANDIDATE for outcome in outcomes
        ),
        "geometric_mean_speedup": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        )
        if speedups
        else None,
        "candidate_workload_time_ms": candidate_time,
        "default_workload_time_ms": default_time,
        "total_workload_speedup": default_time / candidate_time
        if candidate_time
        else None,
        "regression_count": sum(value < 1.0 for value in speedups),
    }


def run_task(
    worker: PostgresClient,
    task_set: TaskSet,
    task: dict[str, Any],
    policy: dict[str, Any],
    agent: QoAgentPolicy | None,
    measurement: RolloutMeasurementSettings,
) -> dict[str, Any]:
    evaluator = RolloutEvaluator(
        worker,
        task_set,
        Task.model_validate(task),
        measurement=measurement,
        max_candidates=MAX_CANDIDATES,
    )
    trace: dict[str, Any] = {}
    try:
        evaluator.start()
        if policy["type"] == PolicyType.RANDOM_STRUCTURED_ACTION:
            action_rng = random.Random(f"{policy['seed']}:{task['task_id']}:actions")
            for _ in range(MAX_CANDIDATES):
                evaluator.evaluate(sample_action(evaluator.catalog, action_rng))
            trace = {"random_seed": policy["seed"]}
        else:
            if agent is None:
                raise RuntimeError("qo-agent policy is not initialized")
            trace = agent.search(evaluator)
        evaluator.finish(random.Random(f"{policy['seed']}:{task['task_id']}:pairs"))
        record = evaluator.record()
    except Exception as error:
        record = evaluator.record(error)
    return {
        "schema_version": 2,
        "status": RunStatus.FAILED.value
        if record.failure
        else RunStatus.COMPLETED.value,
        "completed_at_utc": utc_now(),
        "policy_trace": trace,
        "rollout": record.to_wire(),
    }


def run_task_on_worker(
    pool: ContainerPool,
    task_set: TaskSet,
    task: dict[str, Any],
    policy: dict[str, Any],
    agent: QoAgentPolicy | None,
    measurement: RolloutMeasurementSettings,
) -> tuple[WorkerSlot, dict[str, Any]]:
    with pool.claim_worker() as slot:
        result = run_task(slot.client, task_set, task, policy, agent, measurement)
        result["worker"] = slot.resources.manifest().model_dump()
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


def run_benchmark(
    repository: Path,
    configured: str | None = None,
    *,
    postgres_config_path: Path,
    pool_config_path: Path,
) -> Path:
    postgres_config = PostgresConfig.load(postgres_config_path)
    pool_config = load_pool_config(pool_config_path)
    task_set = TaskSet.load(repository, "job")
    config_path, config = load_run_config(repository, configured)
    policy = config["policy"]
    measurement = RolloutMeasurementSettings.model_validate(config["measurement"])
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
        "inventory_id": task_set.task_set_id,
        "inventory_sha256": sha256_file(task_set.inventory_path),
        "data_identity": {"fixture_id": task_set.fixture_id},
        "runtime_identity": {"postgres_config_id": postgres_config.config_id},
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
            else {
                **agent.manifest(MAX_CANDIDATES),
                "candidate_count": MAX_CANDIDATES,
            }
        ),
        "protocol": {
            "plan_fingerprint_version": PLAN_FINGERPRINT_VERSION,
            **measurement.model_dump(),
        },
        "worker_pool": None,
        "task_count": len(task_set.tasks),
        "completed_task_count": 0,
        "failed_task_count": 0,
        "summary": None,
    }
    manifest_path = output_dir / "run.json"
    write_json(manifest_path, manifest)

    compose_project_name = f"qorl-run-{started_at:%Y%m%d%H%M%S}-{os.getpid()}".lower()
    tasks = [task.model_dump() for task in task_set.tasks]
    results_by_task: dict[str, dict[str, Any]] = {}
    run = TaskRun(
        repository,
        compose_project_name,
        output_dir,
        manifest_path,
        manifest,
        pool_field="worker_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
    )

    def execute_task(
        pool: ContainerPool, task: dict[str, Any]
    ) -> tuple[WorkerSlot, dict[str, Any]]:
        return run_task_on_worker(pool, task_set, task, policy, agent, measurement)

    try:
        with run:
            if run.pool is None:
                raise RuntimeError("benchmark worker pool did not start")
            manifest["protocol"].update(
                {
                    "worker_count": len(run.pool.workers),
                    "concurrent_tasks": min(len(run.pool.workers), len(tasks)),
                    "one_query_per_worker": True,
                }
            )
            run.write()
            for completion in run.map(
                tasks,
                execute_task,
                handled_errors=(ModelError, PostgresError, ContainerError),
            ):
                task = completion.item
                task_id = task["task_id"]
                print(
                    f"[{completion.ordinal}/{manifest['task_count']}] {task_id}",
                    flush=True,
                )
                if completion.error is None:
                    if completion.result is None:
                        raise RuntimeError("benchmark task returned no result")
                    slot, result = completion.result
                    if result["status"] == RunStatus.COMPLETED:
                        manifest["completed_task_count"] += 1
                        print(
                            f"  worker={slot.resources.index} "
                            f"final={result['rollout']['final']['kind']} speedup={result['rollout']['final']['speedup']}"
                        )
                    else:
                        manifest["failed_task_count"] += 1
                        print("  no valid candidate")
                else:
                    result = {
                        "schema_version": 1,
                        "task_id": task["task_id"],
                        "template_id": task["template_id"],
                        "status": RunStatus.FAILED.value,
                        "completed_at_utc": utc_now(),
                        "error": str(completion.error),
                    }
                    manifest["failed_task_count"] += 1
                    print(f"  failed: {completion.error}")
                results_by_task[task_id] = result
                write_json(task_dir / f"{task_id}.json", result)
                run.write()
    except BaseException:
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
