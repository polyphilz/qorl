from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from qorl import __version__
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.calibration import utc_now, write_json
from qorl.fixture import DatabaseFixture, TaskSet, sha256_file
from qorl.pool import WorkerPool, WorkerSlot, start_pool
from qorl.rollout import RolloutEvaluator
from qorl.worker import WorkerError
from scripts.rl.paired_validation import load_tasks, summarize
from scripts.sft.live_protocol_validation import trace_metrics, wait_for_server


CONFIG = Path("experiments/004-rl-run-v2/checkpoint-evaluation.json")
START_POLICY = "start"
PRINT_LOCK = Lock()


def policy_name(step: int) -> str:
    return f"step-{step:03d}"


def load_config(repository: Path) -> tuple[dict[str, Any], Path]:
    path = repository / CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    steps = config.get("checkpoint_steps")
    if (
        config.get("schema_version") != 1
        or config.get("concurrency") != 4
        or not isinstance(steps, list)
        or steps != list(range(10, 101, 10))
    ):
        raise RuntimeError("invalid v2 checkpoint-evaluation configuration")
    return config, path


def checkpoint_adapter(training_run: Path, step: int) -> Path:
    return training_run / "evaluation-adapters" / policy_name(step)


def verify_adapter(path: Path, base_model: Path) -> dict[str, Any]:
    weights = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    manifest_path = path / "qorl-manifest.json"
    if not all(item.is_file() for item in (weights, config_path, manifest_path)):
        raise RuntimeError(f"checkpoint adapter is incomplete: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        config.get("peft_type") != "LORA"
        or config.get("base_model_name_or_path") != str(base_model)
        or manifest.get("adapter_sha256") != sha256_file(weights)
    ):
        raise RuntimeError(f"checkpoint adapter identity differs: {path}")
    return {
        "path": str(path),
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_config_sha256": sha256_file(config_path),
        "tensor_count": manifest["tensor_count"],
        "nonzero_lora_b_values": manifest["nonzero_lora_b_values"],
    }


def prepare_adapters(
    repository: Path,
    base_model: Path,
    training_run: Path,
    steps: list[int],
) -> dict[int, dict[str, Any]]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    identities: dict[int, dict[str, Any]] = {}
    for step in steps:
        checkpoint = training_run / "checkpoints" / f"step_{step}" / "trainer"
        output = checkpoint_adapter(training_run, step)
        if not output.exists():
            if not (checkpoint / ".metadata").is_file():
                raise RuntimeError(f"Prime checkpoint is missing: {checkpoint}")
            subprocess.run(
                [
                    uv,
                    "run",
                    "--project",
                    str(repository / "training"),
                    "--frozen",
                    "--no-sync",
                    "python",
                    str(repository / "training/export_adapter.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--model",
                    str(base_model),
                    "--output",
                    str(output),
                ],
                cwd=repository,
                check=True,
            )
        identities[step] = verify_adapter(output, base_model)
    return identities


def model_command(
    vllm: Path,
    base_model: Path,
    adapters: dict[int, dict[str, Any]],
    policy: dict[str, Any],
    context_length: int,
    concurrency: int,
    port: int,
) -> list[str]:
    command = [
        str(vllm),
        "serve",
        str(base_model),
        "--served-model-name",
        START_POLICY,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--language-model-only",
        "--max-model-len",
        str(context_length),
        "--max-num-seqs",
        str(concurrency),
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
    if adapters:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                "16",
                "--max-loras",
                str(min(concurrency, len(adapters))),
                "--max-cpu-loras",
                str(len(adapters)),
                "--lora-modules",
                *(
                    f"{policy_name(step)}={identity['path']}"
                    for step, identity in adapters.items()
                ),
            ]
        )
    return command


def rotated(values: list[str], offset: int) -> list[str]:
    position = offset % len(values)
    return values[position:] + values[:position]


def checkpoint_summary(
    results: list[dict[str, Any]], planned_rollouts: int
) -> dict[str, Any]:
    summary = summarize(results, planned_rollouts)
    completed = [item for item in results if item["status"] == "completed"]
    valid = [
        item
        for item in completed
        if item["final"]["status"] == "completed"
    ]
    default_winners = [
        item
        for item in valid
        if item["final"].get("winning_plan_sha256")
        == item["default"].get("plan_sha256")
    ]
    summary.update(
        {
            "default_winner_count": len(default_winners),
            "default_winner_rate": (
                len(default_winners) / len(valid) if valid else None
            ),
            "novel_candidate_count": sum(
                result["protocol_metrics"]["novel_candidates"]
                for result in completed
            ),
            "rollout_novel_candidate_rate": (
                sum(
                    result["protocol_metrics"]["has_novel_candidate"]
                    for result in completed
                )
                / len(completed)
                if completed
                else None
            ),
        }
    )
    return summary


def evaluate_once(
    slot: WorkerSlot,
    task_set: TaskSet,
    task: dict[str, Any],
    seed: int,
    model_name: str,
    base_config: QoAgentConfig,
    pair_seed: str,
) -> dict[str, Any]:
    worker = slot.worker
    before = worker.explain_calls, worker.explain_analyze_calls
    evaluator = RolloutEvaluator(worker, task_set, task)
    try:
        baseline = evaluator.start()
        agent = QoAgentPolicy(replace(base_config, model=model_name, seed=seed))
        trace = agent.search(evaluator)
        metrics = trace_metrics(
            evaluator, trace, base_config.maximum_model_turns
        )
        final = evaluator.finish(random.Random(pair_seed))
        result = {
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "task_id": task["task_id"],
            "template_id": task["template_id"],
            "rollout_seed": seed,
            "model": model_name,
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
            "model": model_name,
            "worker_slot": slot.resources.index,
            "error": str(error),
        }
    result["database_calls"] = {
        "explain": worker.explain_calls - before[0],
        "explain_analyze": worker.explain_analyze_calls - before[1],
    }
    return result


def evaluate_series(
    pool: WorkerPool,
    task_set: TaskSet,
    task: dict[str, Any],
    seed: int,
    policies: list[str],
    offset: int,
    base_config: QoAgentConfig,
    evaluation_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with pool.claim_worker() as slot:
        for model_name in rotated(policies, offset):
            with PRINT_LOCK:
                print(
                    f"[{task['task_id']} seed={seed}] {model_name} "
                    f"worker={slot.resources.index}",
                    flush=True,
                )
            result = evaluate_once(
                slot,
                task_set,
                task,
                seed,
                model_name,
                base_config,
                f"{evaluation_id}:{task['task_id']}:{seed}:pairs",
            )
            write_json(
                output_dir
                / model_name
                / "rollouts"
                / f"{task['task_id']}--{seed}.json",
                result,
            )
            results.append(result)
            with PRINT_LOCK:
                if result["status"] == "completed":
                    final = result["final"]
                    label = (
                        f"{final['score']:.3f}x"
                        if final["status"] == "completed"
                        else final["status"]
                    )
                else:
                    label = f"failed: {result['error']}"
                print(f"  {model_name}: {label}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the v2 starting policy and saved checkpoints."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the starting policy and step 100 on one task and one seed",
    )
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config, config_path = load_config(repository)
    base_model = (repository / config["base_model"]).resolve()
    training_run = (repository / config["training_run"]).resolve()
    if not (base_model / "model.safetensors").is_file():
        raise RuntimeError(f"v2 starting model is missing: {base_model}")
    steps = [100] if arguments.preflight else config["checkpoint_steps"]
    adapters = prepare_adapters(repository, base_model, training_run, steps)

    policy_path = repository / config["run_config"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))["policy"]
    vllm = repository / ".venv-vllm/bin/vllm"
    if not vllm.is_file():
        raise RuntimeError(f"pinned evaluation vLLM is missing: {vllm}")

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, config["task_set"], fixture.identity)
    tasks, selection_path = load_tasks(repository, task_set, config)
    seeds = config["rollout_seeds"]
    if arguments.preflight:
        tasks, seeds = tasks[:1], seeds[:1]
    policies = [START_POLICY, *(policy_name(step) for step in steps)]
    expected_rollouts = len(tasks) * len(seeds)
    suffix = "-preflight" if arguments.preflight else ""
    output_dir = (
        repository
        / "outputs/rl"
        / f"{config['evaluation_id']}{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "mode": "preflight" if arguments.preflight else "full",
        "status": "starting",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "config_sha256": sha256_file(config_path),
        "selection_sha256": sha256_file(selection_path),
        "run_config_sha256": sha256_file(policy_path),
        "snapshot_manifest_sha256": sha256_file(
            fixture.snapshot_manifest_path
        ),
        "base_model": {
            "path": config["base_model"],
            "model_sha256": sha256_file(base_model / "model.safetensors"),
        },
        "adapters": {
            policy_name(step): {
                **identity,
                "path": str(Path(identity["path"]).relative_to(repository)),
            }
            for step, identity in adapters.items()
        },
        "orchestrator": {
            "qorl_version": __version__,
            "python_version": platform.python_version(),
            "concurrency": config["concurrency"],
        },
        "task_ids": [task["task_id"] for task in tasks],
        "rollout_seeds": seeds,
        "policies": {
            name: {"status": "pending", "summary": None}
            for name in policies
        },
        "error": None,
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)

    command = model_command(
        vllm,
        base_model,
        adapters,
        policy,
        config["context_length"],
        config["concurrency"],
        arguments.port,
    )
    environment = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    process: subprocess.Popen[Any] | None = None
    pool: WorkerPool | None = None
    results = {name: [] for name in policies}
    with (output_dir / "vllm.log").open("w") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=repository,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            wait_for_server(
                f"http://127.0.0.1:{arguments.port}/health",
                process,
                arguments.startup_timeout,
            )
            base_config = QoAgentConfig.from_dict(
                {
                    **policy,
                    "model": START_POLICY,
                    "base_url": f"http://127.0.0.1:{arguments.port}/v1",
                    "context_length": config["context_length"],
                }
            )
            report["model_server"] = QoAgentPolicy(base_config).preflight()
            pool = start_pool(
                fixture, f"qorl-v2-checkpoints-{os.getpid()}"
            )
            report["database_pool"] = pool.manifest()
            for slot in pool.workers:
                slot.worker.capture_environment(
                    output_dir / "environment" / f"worker-{slot.resources.index}",
                    "pre",
                )
            report["status"] = "running"
            for name in policies:
                report["policies"][name]["status"] = "running"
            write_json(report_path, report)

            jobs = [
                (task, seed, index)
                for index, (task, seed) in enumerate(
                    (task, seed) for task in tasks for seed in seeds
                )
            ]
            with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
                futures = [
                    executor.submit(
                        evaluate_series,
                        pool,
                        task_set,
                        task,
                        seed,
                        policies,
                        index,
                        base_config,
                        config["evaluation_id"],
                        output_dir,
                    )
                    for task, seed, index in jobs
                ]
                for future in as_completed(futures):
                    for result in future.result():
                        results[result["model"]].append(result)
                    for name in policies:
                        report["policies"][name]["summary"] = checkpoint_summary(
                            results[name], expected_rollouts
                        )
                    write_json(report_path, report)

            for slot in pool.workers:
                slot.worker.capture_environment(
                    output_dir / "environment" / f"worker-{slot.resources.index}",
                    "post",
                )
            for name in policies:
                failures = sum(
                    item["status"] != "completed" for item in results[name]
                )
                report["policies"][name]["status"] = (
                    "completed" if not failures else "completed_with_failures"
                )
            report["status"] = (
                "completed"
                if all(
                    item["status"] == "completed"
                    for item in report["policies"].values()
                )
                else "completed_with_failures"
            )
            report["completed_at_utc"] = utc_now()
            write_json(report_path, report)
            print(json.dumps(report["policies"], indent=2), flush=True)
            print(f"checkpoint validation: {output_dir}", flush=True)
        except BaseException as error:
            report["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            report["completed_at_utc"] = utc_now()
            report["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            write_json(report_path, report)
            raise
        finally:
            if pool is not None:
                pool.close()
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
