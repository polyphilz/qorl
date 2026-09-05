from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError
from qorl.agent.types import StopReason
from qorl.db.exceptions import WorkerError
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.measure.run import TaskRun
from qorl.measure.schemas import RunStatus
from qorl.sft.assemble import action_families
from qorl.sft.sample import PlanValidationEvaluator, file_identity
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    JSON_OBJECT_LIST_ADAPTER,
    ActionFamily,
    DatasetConfig,
    DatasetSelection,
    GateReport,
    GateRollout,
    GateSummary,
    JsonObject,
    PipelineError,
    load_json_object,
    load_record,
    require_object,
    require_string,
)
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet


@dataclass(frozen=True)
class GateRequest:
    task: JsonObject
    task_id: str
    template_id: str
    sample: int
    seed: int


def gate_seed(dataset_seed: int, task_id: str, sample: int) -> int:
    digest = hashlib.sha256(
        f"{dataset_seed}:live_gate:{task_id}:{sample}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def evaluate_request(
    pool: WorkerPool,
    task_set: TaskSet,
    request: GateRequest,
    config: QoAgentConfig,
) -> tuple[WorkerSlot, GateRollout]:
    with pool.claim_worker() as slot:
        evaluator = PlanValidationEvaluator(slot.worker, task_set, request.task)
        try:
            evaluator.start()
            trace = JSON_OBJECT_ADAPTER.validate_python(
                QoAgentPolicy(replace(config, seed=request.seed)).search(evaluator)
            )
            candidate = evaluator.candidates[0] if evaluator.candidates else None
            action = (
                JSON_OBJECT_ADAPTER.validate_python(candidate.action)
                if candidate is not None and candidate.action_valid
                else None
            )
            families = (
                [ActionFamily(value) for value in action_families(action)]
                if action is not None
                else []
            )
            kept_default = trace.get("stop_reason") == StopReason.MODEL_KEEP_DEFAULT
            constrained = (
                candidate.constraints_satisfied if candidate is not None else False
            )
            duplicate = (
                candidate.duplicate_of is not None
                if constrained and candidate is not None
                else False
            )
            novel_fingerprint = (
                candidate.plan_sha256
                if constrained and not duplicate and candidate is not None
                else None
            )
            return slot, GateRollout(
                task_id=request.task_id,
                template_id=request.template_id,
                cohort="live_gate",
                sample=request.sample,
                seed=request.seed,
                status=RunStatus.COMPLETED,
                decision="keep_default"
                if kept_default
                else "candidate"
                if candidate is not None
                else None,
                action_valid=candidate.action_valid if candidate is not None else False,
                constraints_satisfied=constrained,
                default_duplicate=duplicate,
                novel_fingerprint=novel_fingerprint,
                action_families=families,
                error=None,
            )
        except (ModelError, WorkerError, RuntimeError) as error:
            return slot, GateRollout(
                task_id=request.task_id,
                template_id=request.template_id,
                cohort="live_gate",
                sample=request.sample,
                seed=request.seed,
                status=RunStatus.FAILED,
                decision=None,
                action_valid=False,
                constraints_satisfied=False,
                default_duplicate=False,
                novel_fingerprint=None,
                action_families=[],
                error=PipelineError(type=type(error).__name__, message=str(error)),
            )


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(records: list[GateRollout]) -> GateSummary:
    completed = [record for record in records if record.status == RunStatus.COMPLETED]
    return GateSummary(
        task_count=len({record.task_id for record in records}),
        rollout_count=len(records),
        completed_rollouts=len(completed),
        failed_rollouts=len(records) - len(completed),
        valid_plan_rate=ratio(
            sum(record.constraints_satisfied for record in completed), len(completed)
        ),
        novel_plan_rate=ratio(
            sum(record.novel_fingerprint is not None for record in completed),
            len(completed),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SFT v2 plan validity and novelty on its live gate."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sft/protocol-sft-train-v2/live-gate.json"),
    )
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config_path = (repository / arguments.config).resolve()
    output = (repository / arguments.output).resolve()
    config = load_record(config_path, DatasetConfig)
    selection_path = (repository / config.selection).resolve()
    selection = load_record(selection_path, DatasetSelection)
    policy_path = (repository / config.policy_config).resolve()
    policy = require_object(load_json_object(policy_path).get("policy"), "policy")
    agent_config = replace(QoAgentConfig.from_dict(policy), model=arguments.model)
    server_identity = JSON_OBJECT_ADAPTER.validate_python(
        QoAgentPolicy(agent_config).preflight()
    )

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb")
    tasks = JSON_OBJECT_LIST_ADAPTER.validate_python(task_set.inventory["tasks"])
    by_id = {
        require_string(task.get("task_id"), "task.task_id"): task for task in tasks
    }
    requests = [
        GateRequest(
            task=by_id[item.task_id],
            task_id=item.task_id,
            template_id=item.template_id,
            sample=sample,
            seed=gate_seed(config.seed, item.task_id, sample),
        )
        for item in selection.splits.live_gate
        for sample in range(1, config.gate.samples_per_task + 1)
    ]
    started = datetime.now(UTC).isoformat()
    provisional = GateReport(
        status=RunStatus.RUNNING,
        started_at_utc=started,
        completed_at_utc=None,
        model=arguments.model,
        server_identity=server_identity,
        selection=file_identity(repository, selection_path),
        dataset_config=file_identity(repository, config_path),
        summary=None,
        database_pool=None,
        rollouts=[],
    )
    report_wire = provisional.to_wire()
    write_json(output, report_wire)
    run = TaskRun(
        fixture,
        f"qorl-sft-v2-gate-{os.getpid()}",
        output.parent,
        output,
        report_wire,
        pool_field="database_pool",
        environment_dir=output.parent / "live-gate-environment",
    )
    records: list[GateRollout] = []

    def execute(
        pool: WorkerPool, request: GateRequest
    ) -> tuple[WorkerSlot, GateRollout]:
        return evaluate_request(pool, task_set, request, agent_config)

    with run:
        for completion in run.map(
            requests, execute, concurrency=config.gate.concurrency
        ):
            if completion.result is None:
                raise RuntimeError("live-gate rollout returned no result")
            _, result = completion.result
            records.append(result)
            report_wire["rollouts"] = [record.to_wire() for record in records]
            run.write()
            print(
                f"[{completion.ordinal}/{len(requests)}] {result.task_id} sample={result.sample} status={result.status.value}",
                flush=True,
            )

    summary = summarize(records)
    final = provisional.model_copy(
        update={
            "status": RunStatus.COMPLETED
            if summary.failed_rollouts == 0
            else RunStatus.COMPLETED_WITH_FAILURES,
            "completed_at_utc": utc_now(),
            "summary": summary,
            "database_pool": require_object(
                report_wire.get("database_pool"), "database pool"
            ),
            "rollouts": records,
        }
    )
    write_json(output, final.to_wire())
    print(json.dumps(final.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
