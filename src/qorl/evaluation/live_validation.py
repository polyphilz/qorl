from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl.adapters.model import adapter_rank, model_snapshot
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.agent.protocol import AgentProtocol
from qorl.agent.types import InspectionExecutor, ToolName
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot, start_pool
from qorl.db.worker import WorkerError
from qorl.evaluation.types import RunStatus
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import Decision
from qorl.util.hashing import sha256_file
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet

BASE_MODEL = "qorl-base"
ADAPTER_MODEL = "qorl-protocol-adapter"
DATASET = Path("outputs/sft/protocol-sft-v1")
TRAINING_RUN = Path("outputs/sft/protocol-sft-train-v1")
CONCURRENCY = 4
EXPECTED_VALIDATION_TASKS = 64


def wait_for_server(url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=10):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise RuntimeError(f"vLLM did not become ready within {timeout} seconds")


def adapter_path(repository: Path) -> Path:
    run_dir = (repository / TRAINING_RUN).resolve()
    report_path = run_dir / "training-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != RunStatus.PASSED:
        raise RuntimeError(f"training report has not passed: {report_path}")
    relative = Path(report["adapter"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("training report contains an unsafe adapter path")
    adapter = (run_dir / relative).resolve()
    if not adapter.is_relative_to(run_dir):
        raise RuntimeError("training report adapter escapes its run directory")
    for filename in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "qorl-manifest.json",
    ):
        if not (adapter / filename).is_file():
            raise RuntimeError(f"trained adapter is missing {filename}: {adapter}")
    return adapter


def validation_tasks(
    repository: Path, task_set: TaskSet
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = repository / DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [
        item["task_id"]
        for item in manifest["demonstrations"]
        if item["partition"] == "validation"
    ]
    by_id = {task["task_id"]: task for task in task_set.inventory["tasks"]}
    if len(ids) != EXPECTED_VALIDATION_TASKS or len(ids) != len(set(ids)):
        raise RuntimeError("protocol dataset does not identify 64 validation tasks")
    if missing := sorted(set(ids) - set(by_id)):
        raise RuntimeError(f"validation tasks are absent from CEB inventory: {missing}")
    tasks = [by_id[task_id] for task_id in ids]
    if any(task["partition"] != "validation" for task in tasks):
        raise RuntimeError("protocol dataset selected a non-validation CEB task")
    return tasks, manifest


def repeated_inspections(trace: dict[str, Any]) -> int:
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for response in trace["model_responses"]:
        message = response.get("choices", [{}])[0].get("message", {})
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            name = function.get("name", "")
            if name in {
                ToolName.EVALUATE_CANDIDATE,
                ToolName.FINISH,
                ToolName.KEEP_DEFAULT,
            }:
                continue
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, sort_keys=True)
            key = (name, arguments)
            repeats += key in seen
            seen.add(key)
    return repeats


def trace_metrics(
    evaluator: RolloutEvaluator[InspectionExecutor],
    trace: dict[str, Any],
    maximum_turns: int,
) -> dict[str, Any]:
    protocol = AgentProtocol.from_evaluator(evaluator, maximum_turns)
    events_by_turn: dict[int, list[dict[str, Any]]] = {}
    for event in trace["tool_events"]:
        events_by_turn.setdefault(event["turn"], []).append(event)

    candidate_count = 0
    available_calls = 0
    valid_tool_calls = 0
    inspection_calls = 0
    valid_inspection_calls = 0
    fake_candidate_ids = 0
    no_tool_calls = 0
    finish_calls = 0
    keep_default_calls = 0
    first_candidate_turn: int | None = None

    for turn, events in sorted(events_by_turn.items()):
        available = protocol.available_tool_names(turn, candidate_count)
        for event in events:
            name = event.get("name")
            if name is None:
                no_tool_calls += 1
                continue
            result = event.get("result")
            top_level_error = (
                result.get("error")
                if isinstance(result, dict) and isinstance(result.get("error"), str)
                else None
            )
            call_available = name in available
            available_calls += call_available
            valid_tool_calls += call_available and top_level_error is None
            if name == ToolName.FINISH:
                finish_calls += 1
            elif name == ToolName.KEEP_DEFAULT:
                keep_default_calls += 1
            elif name == ToolName.EVALUATE_CANDIDATE:
                if first_candidate_turn is None:
                    first_candidate_turn = turn
            else:
                inspection_calls += 1
                valid_inspection_calls += call_available and top_level_error is None
            if (
                name == ToolName.GET_PLAN
                and top_level_error
                and "not issued" in top_level_error
            ):
                fake_candidate_ids += 1
        candidate_count += sum(
            event.get("name") == ToolName.EVALUATE_CANDIDATE
            and isinstance(event.get("result"), dict)
            and "candidate_id" in event["result"]
            for event in events
        )

    candidates = evaluator.candidates
    valid = [item for item in candidates if item.constraints_satisfied]
    novel = [item for item in valid if item.duplicate_of is None]
    tool_calls = sum(
        event.get("name") is not None
        for events in events_by_turn.values()
        for event in events
    )
    return {
        "model_turns": len(trace["model_responses"]),
        "stop_reason": trace["stop_reason"],
        "tool_calls": tool_calls,
        "available_tool_calls": available_calls,
        "valid_tool_calls": valid_tool_calls,
        "no_tool_call_turns": no_tool_calls,
        "inspection_calls": inspection_calls,
        "valid_inspection_calls": valid_inspection_calls,
        "fake_candidate_id_calls": fake_candidate_ids,
        "repeated_inspection_calls": repeated_inspections(trace),
        "finish_calls": finish_calls,
        "keep_default_calls": keep_default_calls,
        "first_candidate_turn": first_candidate_turn,
        "candidate_attempts": len(candidates),
        "action_valid_candidates": sum(item.action_valid for item in candidates),
        "constraint_satisfied_candidates": len(valid),
        "duplicate_candidates": len(valid) - len(novel),
        "novel_candidates": len(novel),
        "has_valid_candidate": bool(valid),
        "has_novel_candidate": bool(novel),
        "normalized_action_sha256s": sorted(
            {
                hashlib.sha256(
                    json.dumps(
                        item.action, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                for item in candidates
                if item.action_valid
            }
        ),
        "novel_plan_sha256s": sorted(
            {item.plan_sha256 for item in novel if item.plan_sha256}
        ),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == RunStatus.COMPLETED]
    metrics = [item["metrics"] for item in completed]
    totals: Counter[str] = Counter()
    additive = (
        "tool_calls",
        "available_tool_calls",
        "valid_tool_calls",
        "no_tool_call_turns",
        "inspection_calls",
        "valid_inspection_calls",
        "fake_candidate_id_calls",
        "repeated_inspection_calls",
        "finish_calls",
        "keep_default_calls",
        "candidate_attempts",
        "action_valid_candidates",
        "constraint_satisfied_candidates",
        "duplicate_candidates",
        "novel_candidates",
    )
    for item in metrics:
        totals.update({key: item[key] for key in additive})
    first_turns = [
        item["first_candidate_turn"]
        for item in metrics
        if item["first_candidate_turn"] is not None
    ]
    return {
        "task_count": len(results),
        "completed_task_count": len(completed),
        "orchestration_failure_count": len(results) - len(completed),
        "tool_call_count": totals["tool_calls"],
        "available_tool_call_rate": ratio(
            totals["available_tool_calls"], totals["tool_calls"]
        ),
        "valid_tool_call_rate": ratio(totals["valid_tool_calls"], totals["tool_calls"]),
        "inspection_argument_validity_rate": ratio(
            totals["valid_inspection_calls"], totals["inspection_calls"]
        ),
        "candidate_attempt_count": totals["candidate_attempts"],
        "action_compile_validity_rate": ratio(
            totals["action_valid_candidates"], totals["candidate_attempts"]
        ),
        "constraint_satisfied_candidate_rate": ratio(
            totals["constraint_satisfied_candidates"],
            totals["candidate_attempts"],
        ),
        "rollout_valid_candidate_rate": ratio(
            sum(item["has_valid_candidate"] for item in metrics), len(results)
        ),
        "rollout_novel_plan_rate": ratio(
            sum(item["has_novel_candidate"] for item in metrics), len(results)
        ),
        "finish_call_rate": ratio(
            sum(item["finish_calls"] > 0 for item in metrics), len(results)
        ),
        "keep_default_call_rate": ratio(
            sum(item["keep_default_calls"] > 0 for item in metrics),
            len(results),
        ),
        "fake_candidate_id_call_count": totals["fake_candidate_id_calls"],
        "repeated_inspection_call_count": totals["repeated_inspection_calls"],
        "duplicate_candidate_count": totals["duplicate_candidates"],
        "unique_normalized_action_count": len(
            {value for item in metrics for value in item["normalized_action_sha256s"]}
        ),
        "unique_novel_plan_count": len(
            {value for item in metrics for value in item["novel_plan_sha256s"]}
        ),
        "median_turn_to_first_candidate": (
            statistics.median(first_turns) if first_turns else None
        ),
        "no_tool_call_turn_count": totals["no_tool_call_turns"],
    }


def evaluate_live_task(
    pool: WorkerPool,
    task_set: TaskSet,
    task: dict[str, Any],
    agent: QoAgentPolicy,
    maximum_model_turns: int,
) -> tuple[WorkerSlot, dict[str, Any]]:
    with pool.claim_worker() as slot:
        evaluator = RolloutEvaluator(slot.worker, task_set, task)
        try:
            baseline = evaluator.start()
            trace = agent.search(evaluator)
            metrics = trace_metrics(evaluator, trace, maximum_model_turns)
            result = {
                "schema_version": 1,
                "status": RunStatus.COMPLETED.value,
                "completed_at_utc": utc_now(),
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "worker_slot": slot.resources.index,
                "default": baseline.to_wire(),
                "candidates": [
                    candidate.to_wire() for candidate in evaluator.candidates
                ],
                "policy_trace": trace,
                "metrics": metrics,
            }
        except (ModelError, WorkerError) as error:
            result = {
                "schema_version": 1,
                "status": RunStatus.FAILED.value,
                "completed_at_utc": utc_now(),
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "worker_slot": slot.resources.index,
                "error": str(error),
                "candidates": [
                    candidate.to_wire() for candidate in evaluator.candidates
                ],
            }
        return slot, result


def evaluate_policy(
    repository: Path,
    pool: WorkerPool,
    task_set: TaskSet,
    tasks: list[dict[str, Any]],
    base_config: QoAgentConfig,
    policy_name: str,
    model_name: str,
    output_dir: Path,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    policy_dir = output_dir / policy_name
    task_dir = policy_dir / "tasks"
    agent = QoAgentPolicy(replace(base_config, model=model_name))
    identity = agent.preflight()
    results: list[dict[str, Any]] = []
    for slot in pool.workers:
        slot.worker.capture_environment(
            policy_dir / "environment" / f"worker-{slot.resources.index}",
            "pre",
        )

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures: dict[Future[tuple[WorkerSlot, dict[str, Any]]], dict[str, Any]] = {
            executor.submit(
                evaluate_live_task,
                pool,
                task_set,
                task,
                agent,
                base_config.maximum_model_turns,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            slot, result = future.result()
            print(
                f"[{policy_name} {index}/{len(tasks)}] {task['task_id']} "
                f"worker={slot.resources.index}",
                flush=True,
            )
            if result["status"] == RunStatus.COMPLETED:
                metrics = result["metrics"]
                terminal = (
                    Decision.KEEP_DEFAULT.value
                    if metrics["keep_default_calls"]
                    else ToolName.FINISH.value
                    if metrics["finish_calls"]
                    else "none"
                )
                print(
                    "  "
                    f"valid={metrics['constraint_satisfied_candidates']}/"
                    f"{metrics['candidate_attempts']} "
                    f"novel={metrics['novel_candidates']} "
                    f"terminal={terminal}",
                    flush=True,
                )
            else:
                print(f"  failed: {result['error']}", flush=True)
            results.append(result)
            write_json(task_dir / f"{task['task_id']}.json", result)
            summary = summarize(results)
            write_json(policy_dir / "summary.json", summary)
            if progress is not None:
                progress(summary)

    for slot in pool.workers:
        slot.worker.capture_environment(
            policy_dir / "environment" / f"worker-{slot.resources.index}",
            "post",
        )

    return {
        "status": RunStatus.COMPLETED.value,
        "model": model_name,
        "server_identity": identity,
        "summary": summarize(results),
    }


def selected_policies(mode: str, order: str) -> list[tuple[str, str]]:
    policies = {
        "adapter": [("adapter", ADAPTER_MODEL)],
        "base": [("base", BASE_MODEL)],
        "both": [("adapter", ADAPTER_MODEL), ("base", BASE_MODEL)],
    }[mode]
    if mode == "both" and order == "base-first":
        policies.reverse()
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base and protocol-SFT policies on held-out CEB tasks."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--policies",
        choices=("adapter", "base", "both"),
        default="adapter",
        help="policies to evaluate (default: adapter)",
    )
    parser.add_argument(
        "--policy-order",
        choices=("adapter-first", "base-first"),
        default="adapter-first",
        help="evaluation order when --policies=both",
    )
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    run_policy = json.loads((repository / "configs/policy/run-v1.json").read_text())[
        "policy"
    ]
    snapshot = model_snapshot(run_policy)
    adapter = adapter_path(repository) if arguments.policies != "base" else None
    vllm = repository / ".venv-vllm/bin/vllm"
    if not vllm.is_file():
        raise RuntimeError(f"pinned evaluation vLLM is missing: {vllm}")

    started = datetime.now(UTC)
    output_dir = (
        repository
        / "outputs/sft"
        / started.strftime("protocol-sft-live-v1-%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "vllm.log"
    base_url = f"http://127.0.0.1:{arguments.port}/v1"
    order = selected_policies(arguments.policies, arguments.policy_order)
    command = [
        str(vllm),
        "serve",
        str(snapshot),
        "--served-model-name",
        BASE_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(arguments.port),
        "--language-model-only",
        "--max-model-len",
        str(run_policy["context_length"]),
        "--max-num-seqs",
        str(CONCURRENCY),
        "--gpu-memory-utilization",
        "0.9",
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        run_policy["tool_call_parser"],
        "--enforce-eager",
    ]
    if adapter is not None:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(adapter_rank(adapter)),
                "--lora-modules",
                f"{ADAPTER_MODEL}={adapter}",
            ]
        )

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    tasks, dataset = validation_tasks(repository, task_set)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": RunStatus.STARTING.value,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None,
        "active_policy": None,
        "protocol": "live CEB protocol evaluation; no final timing pairs",
        "policy_order": [name for name, _ in order],
        "task_set_id": "ceb-v1",
        "data_identity": fixture.data_identity,
        "runtime_identity": fixture.runtime_identity,
        "database_pool": None,
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "dataset_manifest_sha256": sha256_file(repository / DATASET / "manifest.json"),
        "dataset_id": dataset["dataset_id"],
        "adapter_manifest_sha256": (
            sha256_file(adapter / "qorl-manifest.json") if adapter else None
        ),
        "policies": {},
        "error": None,
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)

    environment = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    pool: WorkerPool | None = None
    with log_path.open("w") as log:
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
            config = QoAgentConfig.from_dict(
                {**run_policy, "base_url": base_url, "model": BASE_MODEL}
            )
            pool = start_pool(fixture, f"qorl-ceb-live-{os.getpid()}")
            report["database_pool"] = pool.manifest()
            report["status"] = RunStatus.RUNNING.value
            write_json(report_path, report)
            for policy_name, model_name in order:
                report["active_policy"] = policy_name
                report["policies"][policy_name] = {
                    "status": RunStatus.RUNNING.value,
                    "model": model_name,
                    "summary": None,
                }
                write_json(report_path, report)

                def record_progress(
                    summary: dict[str, Any], name: str = policy_name
                ) -> None:
                    report["policies"][name]["summary"] = summary
                    write_json(report_path, report)

                report["policies"][policy_name] = evaluate_policy(
                    repository,
                    pool,
                    task_set,
                    tasks,
                    config,
                    policy_name,
                    model_name,
                    output_dir,
                    record_progress,
                )
                report["active_policy"] = None
                write_json(report_path, report)
            report["status"] = RunStatus.COMPLETED.value
            report["completed_at_utc"] = utc_now()
            write_json(report_path, report)
            print(json.dumps(report["policies"], indent=2), flush=True)
            print(f"live protocol evaluation: {output_dir}", flush=True)
        except BaseException as error:
            status = (
                RunStatus.INTERRUPTED.value
                if isinstance(error, KeyboardInterrupt)
                else RunStatus.FAILED.value
            )
            active = report["active_policy"]
            if active is not None:
                summary_path = output_dir / active / "summary.json"
                if summary_path.is_file():
                    report["policies"][active]["summary"] = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                report["policies"][active]["status"] = status
            report["status"] = status
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
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
