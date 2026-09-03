from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl import __version__
from qorl.adapters.model import model_snapshot
from qorl.adapters.verify import verify_merged_model
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot, start_pool
from qorl.db.worker import WorkerError
from qorl.evaluation.live_validation import (
    adapter_path,
    trace_metrics,
    wait_for_server,
)
from qorl.measure.rollout import RolloutEvaluator
from qorl.util.hashing import sha256_file
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet

CONFIG = Path("experiments/003-rl-pilot-v1/validation.json")
SERVED_MODEL = "qorl-rl-pilot-policy"
CONCURRENCY = 4


def load_tasks(
    repository: Path,
    task_set: TaskSet,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    selection_path = repository / config["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["splits"][config["split"]]
    tasks = {task["task_id"]: task for task in task_set.inventory["tasks"]}
    chosen = [tasks[item["task_id"]] for item in selected]
    if len(chosen) != 16 or len({task["task_id"] for task in chosen}) != 16:
        raise RuntimeError("paired validation requires 16 unique tasks")
    counts = Counter(task["template_id"] for task in chosen)
    if set(counts.values()) != {4} or len(counts) != 4:
        raise RuntimeError("paired validation requires four tasks from four templates")
    if any(task["partition"] != "validation" for task in chosen):
        raise RuntimeError("paired validation selected a training task")
    return chosen, selection_path


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summarize(
    results: list[dict[str, Any]], planned_rollout_count: int = 64
) -> dict[str, Any]:
    completed = [result for result in results if result["status"] == "completed"]
    finals = [result["final"] for result in completed]
    scored = [final for final in finals if final["status"] == "completed"]
    candidates = [
        candidate for result in completed for candidate in result["candidates"]
    ]
    rewards = [float(final["trajectory_reward"]) for final in finals]
    scores = [float(final["score"]) for final in scored]
    prompt_tokens = sum(
        result["policy_trace"]["usage"].get("prompt_tokens", 0) for result in completed
    )
    output_tokens = sum(
        result["policy_trace"]["usage"].get("completion_tokens", 0)
        for result in completed
    )
    groups: dict[str, dict[str, Any]] = {}
    for task_id in sorted({result["task_id"] for result in completed}):
        members = [result for result in completed if result["task_id"] == task_id]
        group_rewards = [float(item["final"]["trajectory_reward"]) for item in members]
        groups[task_id] = {
            "rollout_count": len(members),
            "mean_reward": statistics.fmean(group_rewards),
            "reward_variance": statistics.pvariance(group_rewards),
            "valid_rollout_count": sum(
                item["final"]["status"] == "completed" for item in members
            ),
        }
    candidate_time = sum(
        float(final["candidate_median_execution_time_ms"]) for final in scored
    )
    default_time = sum(
        float(final["default_median_execution_time_ms"]) for final in scored
    )
    return {
        "planned_rollout_count": planned_rollout_count,
        "completed_rollout_count": len(completed),
        "orchestration_failure_count": len(results) - len(completed),
        "valid_rollout_count": len(scored),
        "valid_rollout_rate": len(scored) / len(completed) if completed else None,
        "candidate_attempt_count": len(candidates),
        "action_valid_candidate_count": sum(
            candidate["action_valid"] for candidate in candidates
        ),
        "constraint_satisfied_candidate_count": sum(
            candidate["constraints_satisfied"] for candidate in candidates
        ),
        "duplicate_candidate_count": sum(
            candidate.get("duplicate_of") is not None for candidate in candidates
        ),
        "unique_action_count": len(
            {fingerprint(candidate["action"]) for candidate in candidates}
        ),
        "unique_novel_plan_count": len(
            {
                candidate["plan_sha256"]
                for candidate in candidates
                if candidate["constraints_satisfied"]
                and candidate.get("duplicate_of") is None
                and candidate.get("plan_sha256")
            }
        ),
        "mean_trajectory_reward": statistics.fmean(rewards) if rewards else None,
        "geometric_mean_speedup": (
            math.exp(statistics.fmean(math.log(score) for score in scores))
            if scores
            else None
        ),
        "total_workload_speedup": (
            default_time / candidate_time if candidate_time else None
        ),
        "regression_count": sum(score < 1.0 for score in scores),
        "prompt_token_count": prompt_tokens,
        "output_token_count": output_tokens,
        "total_token_count": prompt_tokens + output_tokens,
        "postgres_explain_call_count": sum(
            result["database_calls"]["explain"] for result in completed
        ),
        "postgres_explain_analyze_call_count": sum(
            result["database_calls"]["explain_analyze"] for result in completed
        ),
        "task_group_count": len(groups),
        "nonzero_reward_variance_group_count": sum(
            group["reward_variance"] > 0 for group in groups.values()
        ),
        "task_groups": groups,
    }


def model_command(
    vllm: Path,
    model: Path,
    policy: dict[str, Any],
    context_length: int,
    port: int,
) -> list[str]:
    return [
        str(vllm),
        "serve",
        str(model),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--language-model-only",
        "--max-model-len",
        str(context_length),
        "--max-num-seqs",
        str(CONCURRENCY),
        "--gpu-memory-utilization",
        "0.9",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        policy["tool_call_parser"],
        "--enforce-eager",
    ]


def evaluate_rollout(
    pool: WorkerPool,
    task_set: TaskSet,
    task: dict[str, Any],
    seed: int,
    base_config: QoAgentConfig,
    evaluation_id: str,
) -> tuple[WorkerSlot, dict[str, Any]]:
    with pool.claim_worker() as slot:
        worker = slot.worker
        before = worker.explain_calls, worker.explain_analyze_calls
        evaluator = RolloutEvaluator(worker, task_set, task)
        try:
            baseline = evaluator.start()
            agent = QoAgentPolicy(replace(base_config, seed=seed))
            trace = agent.search(evaluator)
            metrics = trace_metrics(
                evaluator,
                trace,
                base_config.maximum_model_turns,
            )
            final = evaluator.finish(
                random.Random(f"{evaluation_id}:{task['task_id']}:{seed}:pairs")
            )
            result = {
                "schema_version": 1,
                "status": "completed",
                "completed_at_utc": utc_now(),
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "rollout_seed": seed,
                "worker_slot": slot.resources.index,
                "default": baseline,
                "candidates": evaluator.candidates,
                "final": final,
                "policy_trace": trace,
                "protocol_metrics": metrics,
            }
        except (ModelError, WorkerError) as error:
            result = {
                "schema_version": 1,
                "status": "failed",
                "completed_at_utc": utc_now(),
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "rollout_seed": seed,
                "worker_slot": slot.resources.index,
                "error": str(error),
            }
        result["database_calls"] = {
            "explain": worker.explain_calls - before[0],
            "explain_analyze": worker.explain_analyze_calls - before[1],
        }
        return slot, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen paired CEB measurement before or after RL."
    )
    parser.add_argument("phase", choices=("pre", "post"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config_path = repository / CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("unknown paired-validation configuration")
    model = arguments.model
    if model is None:
        if arguments.phase != "pre":
            raise RuntimeError("post-RL validation requires --model")
        model = Path(config["pre_rl_model"])
    model = model if model.is_absolute() else repository / model
    if not model.is_dir():
        raise RuntimeError(f"validation model is missing: {model}")

    policy_path = repository / config["run_config"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))["policy"]

    model_sha256: str
    if arguments.phase == "pre":
        base = model_snapshot(policy)
        verify_merged_model(base, adapter_path(repository), model)
        merge_manifest_path = model / "qorl-merge.json"
        merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
        model_sha256 = merge_manifest["merged_model_sha256"]
    else:
        merge_manifest_path = None
        model_sha256 = sha256_file(model / "model.safetensors")

    vllm = repository / ".venv-vllm/bin/vllm"
    if not vllm.is_file():
        raise RuntimeError(f"pinned evaluation vLLM is missing: {vllm}")

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, config["task_set"], fixture.data_identity)
    tasks, selection_path = load_tasks(repository, task_set, config)
    seeds = config["rollout_seeds"]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise RuntimeError("paired validation requires four unique rollout seeds")

    output_dir = repository / "outputs/rl" / config["evaluation_id"] / arguments.phase
    output_dir.mkdir(parents=True, exist_ok=False)
    rollout_dir = output_dir / "rollouts"
    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "phase": arguments.phase,
        "status": "starting",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None,
        "config_sha256": sha256_file(config_path),
        "selection_sha256": sha256_file(selection_path),
        "run_config_sha256": sha256_file(policy_path),
        "snapshot_manifest_sha256": sha256_file(fixture.snapshot_manifest_path),
        "data_identity": fixture.data_identity,
        "runtime_identity": fixture.runtime_identity,
        "model": {
            "path": str(model.relative_to(repository)),
            "model_safetensors_sha256": model_sha256,
            "merge_manifest_sha256": (
                sha256_file(merge_manifest_path) if merge_manifest_path else None
            ),
        },
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
        },
        "task_ids": [task["task_id"] for task in tasks],
        "rollout_seeds": seeds,
        "summary": None,
        "error": None,
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)

    command = model_command(
        vllm, model, policy, config["context_length"], arguments.port
    )
    environment = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    results: list[dict[str, Any]] = []
    pool: WorkerPool | None = None
    with (output_dir / "vllm.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            wait_for_server(
                f"http://127.0.0.1:{arguments.port}/health",
                process,
                arguments.startup_timeout,
            )
            base_config = QoAgentConfig.from_dict(
                {
                    **policy,
                    "model": SERVED_MODEL,
                    "base_url": f"http://127.0.0.1:{arguments.port}/v1",
                    "context_length": config["context_length"],
                }
            )
            identity = QoAgentPolicy(base_config).preflight()
            report["model_server"] = identity
            pool = start_pool(
                fixture,
                f"qorl-rl-validation-{arguments.phase}-{os.getpid()}",
            )
            report["database_pool"] = pool.manifest()
            report["orchestrator"]["concurrency"] = CONCURRENCY
            for slot in pool.workers:
                slot.worker.capture_environment(
                    output_dir / "environment" / f"worker-{slot.resources.index}",
                    "pre",
                )
            report["status"] = "running"
            write_json(report_path, report)

            jobs = [(task, seed) for task in tasks for seed in seeds]
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                futures: dict[
                    Future[tuple[WorkerSlot, dict[str, Any]]],
                    tuple[dict[str, Any], int],
                ] = {
                    executor.submit(
                        evaluate_rollout,
                        pool,
                        task_set,
                        task,
                        seed,
                        base_config,
                        config["evaluation_id"],
                    ): (task, seed)
                    for task, seed in jobs
                }
                for ordinal, future in enumerate(as_completed(futures), start=1):
                    task, seed = futures[future]
                    slot, result = future.result()
                    print(
                        f"[{ordinal}/{len(jobs)}] {task['task_id']} seed={seed} "
                        f"worker={slot.resources.index}",
                        flush=True,
                    )
                    if result["status"] == "completed":
                        final = result["final"]
                        label = (
                            f"{final['score']:.3f}x "
                            f"reward={final['trajectory_reward']:.3f}"
                            if final["status"] == "completed"
                            else "no valid candidate "
                            f"reward={final['trajectory_reward']:.3f}"
                        )
                    else:
                        label = f"failed: {result['error']}"
                    print(f"  final={label}", flush=True)
                    results.append(result)
                    filename = f"{task['task_id']}--{seed}.json"
                    write_json(rollout_dir / filename, result)
                    report["summary"] = summarize(results)
                    write_json(report_path, report)

            for slot in pool.workers:
                slot.worker.capture_environment(
                    output_dir / "environment" / f"worker-{slot.resources.index}",
                    "post",
                )
            pool.close()
            pool = None

            report["status"] = (
                "completed"
                if all(result["status"] == "completed" for result in results)
                else "completed_with_failures"
            )
            report["completed_at_utc"] = utc_now()
            report["summary"] = summarize(results)
            write_json(report_path, report)
            print(json.dumps(report["summary"], indent=2), flush=True)
            print(f"paired validation: {output_dir}", flush=True)
        except BaseException as error:
            report["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            report["completed_at_utc"] = utc_now()
            report["summary"] = summarize(results)
            report["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            write_json(report_path, report)
            raise
        finally:
            if pool is not None:
                pool.close()
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
