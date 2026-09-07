from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.types import StopReason
from qorl.measure.run import TaskRun
from qorl.measure.schemas import RunStatus
from qorl.measure.timeouts import GLOBAL_TIMEOUT_MS, CalibratedTimeouts
from qorl.measure.validation import PlanValidationEvaluator
from qorl.paths import REPOSITORY_ROOT
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.postgres.config import PostgresConfig
from qorl.sft.assemble import action_families
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    JSON_OBJECT_LIST_ADAPTER,
    ActionFamily,
    DatasetConfig,
    DatasetSelection,
    FallbackCheck,
    FileIdentity,
    JsonObject,
    PipelineError,
    SampleRecord,
    SamplerIdentity,
    SamplingInvocation,
    SamplingManifest,
    SamplingMode,
    SamplingSummary,
    load_json_object,
    load_record,
    load_string_list,
    require_list,
    require_object,
    require_string,
)
from qorl.taskset.schemas import Task
from qorl.taskset.taskset import TaskSet
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json
from qorl.util.time import utc_now
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import WorkerSlot

SAMPLING_ID = "qorl-protocol-sft-v2-sampling-v2"


@dataclass(frozen=True)
class SampleRequest:
    task: JsonObject
    task_id: str
    template_id: str
    sample: int
    seed: int
    sampling_mode: SamplingMode


def sample_seed(dataset_seed: int, task_id: str, sample: int) -> int:
    digest = hashlib.sha256(
        f"{dataset_seed}:{task_id}:sample:{sample}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def display_path(repository: Path, path: Path) -> str:
    try:
        return path.relative_to(repository).as_posix()
    except ValueError:
        return str(path)


def selected_tasks(
    task_set: TaskSet, selection: DatasetSelection, split: str
) -> list[JsonObject]:
    try:
        selected = getattr(selection.splits, split)
    except AttributeError as error:
        raise RuntimeError(f"selection has no {split!r} split") from error
    tasks = JSON_OBJECT_LIST_ADAPTER.validate_python(
        [task.model_dump() for task in task_set.tasks]
    )
    by_id = {
        require_string(task.get("task_id"), "task.task_id"): task for task in tasks
    }
    task_ids = [item.task_id for item in selected]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("selection contains duplicate task IDs")
    try:
        return [by_id[task_id] for task_id in task_ids]
    except KeyError as error:
        raise RuntimeError(f"selection contains an unknown task: {error}") from error


def training_transcript(trace: JsonObject) -> list[JsonObject]:
    return [
        require_object(item, "policy_trace.transcript[]")
        for item in require_list(trace.get("transcript"), "policy_trace.transcript")
    ]


def evaluate_request(
    pool: ContainerPool,
    task_set: TaskSet,
    request: SampleRequest,
    config: QoAgentConfig,
    calibrated_timeouts: CalibratedTimeouts | None,
    sampler_identity: SamplerIdentity,
) -> tuple[WorkerSlot, SampleRecord]:
    with pool.claim_worker() as slot:
        timeout = (
            calibrated_timeouts.task(request.task_id)
            if calibrated_timeouts is not None
            else None
        )
        evaluator = PlanValidationEvaluator(
            slot.client,
            task_set,
            Task.model_validate(request.task),
            default_timeout_ms=timeout.timeout_ms if timeout else GLOBAL_TIMEOUT_MS,
            max_candidates=1,
        )
        try:
            baseline = evaluator.start()
            raw_trace = QoAgentPolicy(replace(config, seed=request.seed)).search(
                evaluator
            )
            trace = JSON_OBJECT_ADAPTER.validate_python(raw_trace)
            status = RunStatus.COMPLETED
            error = None
        except Exception as caught:
            baseline = evaluator.default
            trace = None
            status = RunStatus.FAILED
            error = PipelineError(type=type(caught).__name__, message=str(caught))
        return slot, SampleRecord(
            status=status,
            completed_at_utc=utc_now(),
            task_id=request.task_id,
            template_id=request.template_id,
            sample=request.sample,
            seed=request.seed,
            sampling_mode=request.sampling_mode,
            steered=False,
            guidance=None,
            worker=JSON_OBJECT_ADAPTER.validate_python(
                slot.resources.manifest().model_dump()
            ),
            data_identity={"fixture_id": task_set.fixture_id},
            runtime_identity={"postgres_config_id": pool.postgres_config.config_id},
            sampler=sampler_identity,
            default=baseline,
            candidates=evaluator.candidates,
            policy_trace=trace,
            training_transcript=training_transcript(trace)
            if trace is not None
            else None,
            error=error,
        )


def sample_path(output: Path, split: str, request: SampleRequest) -> Path:
    return (
        output
        / "samples"
        / split
        / request.task_id
        / f"sample-{request.sample:02d}.json"
    )


def load_samples(output: Path, split: str) -> list[SampleRecord]:
    return [
        load_record(path, SampleRecord)
        for path in sorted((output / "samples" / split).glob("*/sample-*.json"))
    ]


def sampling_summary(records: list[SampleRecord]) -> SamplingSummary:
    completed = [record for record in records if record.status == RunStatus.COMPLETED]
    fingerprints: dict[str, set[str]] = {}
    intervened_tasks: set[str] = set()
    decisions: dict[str, set[str]] = {}
    family_counts: Counter[ActionFamily] = Counter()
    action_valid = constrained = duplicates = novel = keep_default = 0
    leading_attempts = leading_constrained = 0
    for record in completed:
        kept_default = (
            record.policy_trace is not None
            and record.policy_trace.get("stop_reason") == StopReason.MODEL_KEEP_DEFAULT
        )
        if kept_default:
            keep_default += 1
            decisions.setdefault(record.task_id, set()).add("keep_default")
        if record.candidates:
            intervened_tasks.add(record.task_id)
            decisions.setdefault(record.task_id, set()).add("candidate")
        for candidate in record.candidates:
            families: list[ActionFamily] = []
            if candidate.action_valid:
                action_valid += 1
                action = JSON_OBJECT_ADAPTER.validate_python(candidate.action)
                families = [ActionFamily(value) for value in action_families(action)]
                family_counts.update(families)
                if ActionFamily.LEADING in families:
                    leading_attempts += 1
                    leading_constrained += candidate.constraints_satisfied
            if not candidate.constraints_satisfied:
                continue
            constrained += 1
            if candidate.structural_duplicate_of is not None:
                duplicates += 1
                continue
            novel += 1
            if candidate.structural_plan_sha256 is None:
                raise RuntimeError("novel candidate has no plan fingerprint")
            fingerprints.setdefault(record.task_id, set()).add(
                candidate.structural_plan_sha256
            )
    distinct = sum(len(values) for values in fingerprints.values())
    denominator = len(intervened_tasks)
    return SamplingSummary(
        rollouts=len(records),
        completed_rollouts=len(completed),
        failed_rollouts=len(records) - len(completed),
        intervention_rollouts=sum(bool(record.candidates) for record in completed),
        keep_default_rollouts=keep_default,
        action_valid_candidates=action_valid,
        constraint_satisfied_candidates=constrained,
        default_duplicate_candidates=duplicates,
        novel_candidates=novel,
        distinct_novel_fingerprints=distinct,
        distinct_novel_fingerprint_yield=distinct / len(records) if records else 0.0,
        distinct_fingerprints_per_intervened_task=distinct / denominator
        if denominator
        else 0.0,
        mixed_decision_task_share=(
            sum(len(values) > 1 for values in decisions.values()) / len(decisions)
            if decisions
            else 0.0
        ),
        leading_attempts=leading_attempts,
        leading_constraint_satisfied_candidates=leading_constrained,
        action_families=dict(sorted(family_counts.items())),
    )


def file_identity(repository: Path, path: Path) -> FileIdentity:
    return FileIdentity(path=display_path(repository, path), sha256=sha256_file(path))


def sample_limit(
    config: DatasetConfig,
    mode: SamplingMode,
    sample_start: int,
    sample_count: int,
) -> int:
    if sample_start < 1 or sample_count < 1:
        raise RuntimeError("sample range must be positive")
    maximum = (
        config.sampling.default_best_search_maximum_samples_per_task
        if mode == SamplingMode.DEFAULT_BEST
        else config.sampling.normal_maximum_samples_per_task
    )
    if (
        mode == SamplingMode.DEFAULT_BEST
        and sample_start <= config.sampling.normal_maximum_samples_per_task
    ):
        raise RuntimeError(
            "default-best sampling must begin after the normal sample cap"
        )
    sample_end = sample_start + sample_count - 1
    if sample_end > maximum:
        raise RuntimeError(
            f"{mode.value} sampling ends at {sample_end}, above its "
            f"{maximum}-sample cap"
        )
    return sample_end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample one-candidate CEB trajectories for protocol SFT v2."
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--postgres-config", type=Path, required=True)
    parser.add_argument("--pool-config", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--timeouts",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/timeouts.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sft/protocol-sft-v2")
    )
    parser.add_argument("--split", choices=("sampling", "validation"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sampler-manifest", type=Path, required=True)
    parser.add_argument("--sample-start", type=int, default=1)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--default-best", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-ids", type=Path)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    postgres_config = PostgresConfig.load(arguments.postgres_config)
    pool_config = load_pool_config(arguments.pool_config)
    config_path = (repository / arguments.config).resolve()
    output = (repository / arguments.output).resolve()
    timeout_path = (repository / arguments.timeouts).resolve()
    sampler_manifest_path = (repository / arguments.sampler_manifest).resolve()
    config = load_record(config_path, DatasetConfig)
    selection_path = (repository / config.selection).resolve()
    policy_path = (repository / config.policy_config).resolve()
    selection = load_record(selection_path, DatasetSelection)
    policy = require_object(load_json_object(policy_path).get("policy"), "policy")
    sampling_mode = (
        SamplingMode.DEFAULT_BEST if arguments.default_best else SamplingMode.NORMAL
    )
    sample_count = (
        arguments.sample_count
        if arguments.sample_count is not None
        else config.sampling.initial_samples_per_task
    )
    sample_end = sample_limit(
        config, sampling_mode, arguments.sample_start, sample_count
    )
    if arguments.default_best and arguments.split != "sampling":
        raise RuntimeError("default-best search is limited to the sampling split")

    task_set = TaskSet.load(repository, "ceb")
    tasks = selected_tasks(task_set, selection, arguments.split)
    fallback_tasks = tasks[: config.sampling.fallback_check_task_count]
    if arguments.task_ids is not None:
        requested = set(load_string_list((repository / arguments.task_ids).resolve()))
        if not requested:
            raise RuntimeError("--task-ids must not be empty")
        tasks = [
            task
            for task in tasks
            if require_string(task.get("task_id"), "task.task_id") in requested
        ]
        if len(tasks) != len(requested):
            raise RuntimeError("--task-ids contains a task outside the selected split")
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise RuntimeError("--limit must be positive")
        tasks = tasks[: arguments.limit]
    calibrated_timeouts = (
        CalibratedTimeouts.load(
            repository,
            timeout_path,
            task_set,
            postgres_config.config_id,
        )
        if arguments.split == "sampling"
        else None
    )
    sampler_manifest = load_json_object(sampler_manifest_path)
    require_list(sampler_manifest.get("artifacts"), "sampler.artifacts")
    base_config = replace(QoAgentConfig.from_dict(policy), model=arguments.model)
    sampler_identity = SamplerIdentity(
        model=arguments.model,
        manifest_sha256=sha256_file(sampler_manifest_path),
        server_identity=JSON_OBJECT_ADAPTER.validate_python(
            QoAgentPolicy(base_config).preflight()
        ),
    )
    requests = [
        SampleRequest(
            task=task,
            task_id=require_string(task.get("task_id"), "task.task_id"),
            template_id=require_string(task.get("template_id"), "task.template_id"),
            sample=sample,
            seed=sample_seed(
                config.seed, require_string(task.get("task_id"), "task.task_id"), sample
            ),
            sampling_mode=sampling_mode,
        )
        for task in tasks
        for sample in range(arguments.sample_start, sample_end + 1)
    ]
    pending = [
        request
        for request in requests
        if not sample_path(output, arguments.split, request).is_file()
    ]
    manifest_name = (
        "default-best-manifest.json"
        if sampling_mode == SamplingMode.DEFAULT_BEST
        else f"{arguments.split}-manifest.json"
    )
    manifest_path = output / "sampling" / manifest_name
    previous = (
        load_record(manifest_path, SamplingManifest)
        if manifest_path.is_file()
        else None
    )
    if previous is not None and (
        previous.sampling_id != SAMPLING_ID
        or previous.plan_fingerprint_version != PLAN_FINGERPRINT_VERSION
    ):
        raise RuntimeError("sampling identity changed; use a new output directory")
    if previous is not None and (
        previous.sampler.model != sampler_identity.model
        or previous.sampler.manifest_sha256 != sampler_identity.manifest_sha256
    ):
        raise RuntimeError("sampling output already belongs to a different sampler")
    identities = {
        "selection": file_identity(repository, selection_path),
        "dataset_config": file_identity(repository, config_path),
        "policy_config": file_identity(repository, policy_path),
    }
    if previous is not None and any(
        getattr(previous, name).sha256 != identity.sha256
        for name, identity in identities.items()
    ):
        raise RuntimeError("sampling output belongs to different experiment inputs")
    records = load_samples(output, arguments.split)
    manifest = SamplingManifest(
        sampling_id=SAMPLING_ID,
        status=RunStatus.RUNNING,
        started_at_utc=previous.started_at_utc
        if previous
        else datetime.now(UTC).isoformat(),
        completed_at_utc=None,
        split=arguments.split,
        selection=identities["selection"],
        dataset_config=identities["dataset_config"],
        policy_config=identities["policy_config"],
        sampler=sampler_identity,
        sampler_manifest_path=display_path(repository, sampler_manifest_path),
        sampler_manifest=sampler_manifest,
        guidance=None,
        data_identity={"fixture_id": task_set.fixture_id},
        runtime_identity={"postgres_config_id": postgres_config.config_id},
        plan_fingerprint_version=PLAN_FINGERPRINT_VERSION,
        database_pool=previous.database_pool if previous else None,
        summary=sampling_summary(records),
        fallback_check=previous.fallback_check if previous else None,
        invocations=list(previous.invocations) if previous else [],
    )
    manifest_wire = manifest.to_wire()
    write_json(manifest_path, manifest_wire)
    run = TaskRun(
        repository,
        f"qorl-sft-v2-sample-{os.getpid()}",
        output,
        manifest_path,
        manifest_wire,
        pool_field="database_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
        environment_dir=output / "sampling" / f"{arguments.split}-environment",
    )

    def execute(
        pool: ContainerPool, request: SampleRequest
    ) -> tuple[WorkerSlot, SampleRecord]:
        return evaluate_request(
            pool, task_set, request, base_config, calibrated_timeouts, sampler_identity
        )

    with run:
        for completion in run.map(
            pending, execute, concurrency=config.sampling.concurrency
        ):
            if completion.result is None:
                raise RuntimeError("sampling request returned no result")
            _, result = completion.result
            write_json(
                sample_path(output, arguments.split, completion.item), result.to_wire()
            )
            records.append(result)
            manifest_wire["summary"] = sampling_summary(records).to_wire()
            run.write()
            print(
                f"[{completion.ordinal}/{len(pending)}] {result.task_id} sample={result.sample} status={result.status.value}",
                flush=True,
            )

    all_records = load_samples(output, arguments.split)
    requested_keys = {(request.task_id, request.sample) for request in requests}
    invocation_summary = sampling_summary(
        [
            record
            for record in all_records
            if (record.task_id, record.sample) in requested_keys
        ]
    )
    completed_at = utc_now()
    fallback = previous.fallback_check if previous else None
    if (
        arguments.split == "sampling"
        and sampling_mode == SamplingMode.NORMAL
        and arguments.sample_start == 1
        and sample_count == config.sampling.initial_samples_per_task
        and len(tasks) == config.sampling.fallback_check_task_count
        and [request.task_id for request in requests[::sample_count]]
        == [
            require_string(task.get("task_id"), "task.task_id")
            for task in fallback_tasks
        ]
    ):
        threshold = config.sampling.fallback_yield_floor
        fallback = FallbackCheck(
            threshold=threshold,
            passed=invocation_summary.distinct_novel_fingerprint_yield >= threshold,
            summary=invocation_summary,
        )
    invocation = SamplingInvocation(
        completed_at_utc=completed_at,
        task_ids=[
            require_string(task.get("task_id"), "task.task_id") for task in tasks
        ],
        sample_start=arguments.sample_start,
        sample_count=sample_count,
        sampling_mode=sampling_mode,
        sampler=sampler_identity,
        guidance=None,
        summary=invocation_summary,
    )
    final = manifest.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "completed_at_utc": completed_at,
            "database_pool": require_object(
                manifest_wire.get("database_pool"), "database pool"
            ),
            "summary": sampling_summary(all_records),
            "fallback_check": fallback,
            "invocations": [*manifest.invocations, invocation],
        }
    )
    write_json(manifest_path, final.to_wire())
    print(
        json.dumps(
            {
                "invocation": invocation_summary.to_wire(),
                "overall": final.summary.to_wire(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
