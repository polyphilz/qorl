from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
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
    GateChecks,
    GateCohortSummary,
    GateReport,
    GateRollout,
    JsonObject,
    MeasurementManifest,
    PipelineError,
    SamplingManifest,
    TaskLabel,
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
    cohort: str
    sample: int
    seed: int


def gate_seed(dataset_seed: int, cohort: str, task_id: str, sample: int) -> int:
    digest = hashlib.sha256(
        f"{dataset_seed}:{cohort}:{task_id}:gate:{sample}".encode()
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
                cohort=request.cohort,
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
                cohort=request.cohort,
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


def summarize(records: list[GateRollout]) -> GateCohortSummary:
    completed = [record for record in records if record.status == RunStatus.COMPLETED]
    interventions = [record for record in completed if record.decision == "candidate"]
    constrained = [record for record in interventions if record.constraints_satisfied]
    novel = [record for record in constrained if not record.default_duplicate]
    fingerprints: dict[str, set[str]] = defaultdict(set)
    decisions: dict[str, set[str]] = defaultdict(set)
    families: Counter[ActionFamily] = Counter()
    for record in completed:
        if record.decision is not None:
            decisions[record.task_id].add(record.decision)
        if record.novel_fingerprint is not None:
            fingerprints[record.task_id].add(record.novel_fingerprint)
        families.update(record.action_families)
    leading = [
        record
        for record in interventions
        if ActionFamily.LEADING in record.action_families
    ]
    intervened_tasks = {record.task_id for record in interventions}
    return GateCohortSummary(
        task_count=len({record.task_id for record in records}),
        rollout_count=len(records),
        completed_rollouts=len(completed),
        intervention_rate=ratio(len(interventions), len(completed)),
        abstention_rate=ratio(
            sum(record.decision == "keep_default" for record in completed),
            len(completed),
        ),
        action_valid_rate=ratio(
            sum(record.action_valid for record in interventions), len(interventions)
        ),
        constraint_satisfied_rate=ratio(len(constrained), len(interventions)),
        default_duplicate_rate=ratio(
            sum(record.default_duplicate for record in constrained), len(constrained)
        ),
        novel_candidate_rate=ratio(len(novel), len(completed)),
        fingerprints_per_intervened_task=ratio(
            sum(len(values) for values in fingerprints.values()), len(intervened_tasks)
        ),
        mixed_decision_task_share=ratio(
            sum(len(values) > 1 for values in decisions.values()), len(decisions)
        ),
        leading_constraint_satisfied_rate=(
            ratio(sum(record.constraints_satisfied for record in leading), len(leading))
            if leading
            else None
        ),
        action_families=dict(sorted(families.items())),
    )


def labeled_rate(
    records: list[GateRollout],
    labels: dict[str, TaskLabel],
    label: TaskLabel,
    decision: str,
) -> float | None:
    selected = [
        record
        for record in records
        if labels.get(record.task_id) == label and record.status == RunStatus.COMPLETED
    ]
    return (
        ratio(sum(record.decision == decision for record in selected), len(selected))
        if selected
        else None
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the protocol SFT v2 live acceptance gate."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/sft/protocol-sft-v2")
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
    dataset = (repository / arguments.dataset).resolve()
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
    sampling_manifest = load_record(
        dataset / "sampling/sampling-manifest.json", SamplingManifest
    )
    measurement_manifest = load_record(
        dataset / "measurement.json", MeasurementManifest
    )
    if measurement_manifest.task_labels is None:
        raise RuntimeError("measurement manifest has no task labels")

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    tasks = JSON_OBJECT_LIST_ADAPTER.validate_python(task_set.inventory["tasks"])
    by_id = {
        require_string(task.get("task_id"), "task.task_id"): task for task in tasks
    }
    cohort_ids = {
        "validation": [item.task_id for item in selection.splits.validation],
        "unlabeled": [item.task_id for item in selection.splits.live_gate],
        "labeled": sorted(
            task_id
            for task_id, label in measurement_manifest.task_labels.items()
            if label in {TaskLabel.KNOWN_WIN, TaskLabel.DEFAULT_BEST}
        ),
    }
    requests = [
        GateRequest(
            task=by_id[task_id],
            task_id=task_id,
            template_id=require_string(
                by_id[task_id].get("template_id"), "task.template_id"
            ),
            cohort=cohort,
            sample=sample,
            seed=gate_seed(config.seed, cohort, task_id, sample),
        )
        for cohort, task_ids in cohort_ids.items()
        for task_id in task_ids
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
        sampler_reference=sampling_manifest.summary,
        cohorts={},
        labeled_known_win_intervention_rate=None,
        labeled_default_best_abstention_rate=None,
        checks=None,
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
                f"[{completion.ordinal}/{len(requests)}] {result.cohort} {result.task_id} sample={result.sample} status={result.status.value}",
                flush=True,
            )

    cohorts = {
        cohort: summarize([record for record in records if record.cohort == cohort])
        for cohort in cohort_ids
    }
    evaluated = [cohorts["validation"], cohorts["unlabeled"]]
    sampler_novel_rate = ratio(
        sampling_manifest.summary.novel_candidates,
        sampling_manifest.summary.completed_rollouts,
    )
    unlabeled_interventions = sum(
        record.cohort == "unlabeled"
        and record.status == RunStatus.COMPLETED
        and record.decision == "candidate"
        for record in records
    )
    checks = GateChecks(
        constraint_satisfied=all(
            item.constraint_satisfied_rate
            >= config.gate.constraint_satisfied_rate_floor
            for item in evaluated
        ),
        default_duplicates=all(
            item.default_duplicate_rate <= config.gate.default_duplicate_rate_ceiling
            for item in evaluated
        ),
        novel_candidate_improvement=all(
            item.novel_candidate_rate
            >= sampler_novel_rate + config.gate.novel_candidate_rate_improvement
            for item in evaluated
        ),
        unlabeled_intervention=cohorts["unlabeled"].intervention_rate
        >= config.gate.unlabeled_intervention_rate_floor,
        unlabeled_diversity=cohorts["unlabeled"].fingerprints_per_intervened_task
        >= config.gate.fingerprints_per_intervened_task_floor,
        family_coverage=all(
            ratio(
                cohorts["unlabeled"].action_families.get(family, 0),
                unlabeled_interventions,
            )
            >= config.gate.action_family_rate_floor
            for family in config.gate.required_action_families
        ),
    )
    passed = all(
        (
            checks.constraint_satisfied,
            checks.default_duplicates,
            checks.novel_candidate_improvement,
            checks.unlabeled_intervention,
            checks.unlabeled_diversity,
            checks.family_coverage,
        )
    )
    final = provisional.model_copy(
        update={
            "status": RunStatus.PASSED if passed else RunStatus.FAILED,
            "completed_at_utc": utc_now(),
            "cohorts": cohorts,
            "labeled_known_win_intervention_rate": labeled_rate(
                records,
                measurement_manifest.task_labels,
                TaskLabel.KNOWN_WIN,
                "candidate",
            ),
            "labeled_default_best_abstention_rate": labeled_rate(
                records,
                measurement_manifest.task_labels,
                TaskLabel.DEFAULT_BEST,
                "keep_default",
            ),
            "checks": checks,
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
