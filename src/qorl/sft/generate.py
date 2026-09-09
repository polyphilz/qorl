"""Schedule hosted qo-agent conversations and save evidence before student preparation."""

import random
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from threading import BoundedSemaphore, Event
from uuid import uuid4

from qorl.agent.agent import QoAgentPolicy
from qorl.agent.prompts import system_prompt
from qorl.experiment.schemas import SftExperimentConfig
from qorl.measure.environment import capture_environment
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import (
    RunStatus,
)
from qorl.measure.timeouts import seconds_to_ms
from qorl.measure.validation import PlanValidationEvaluator
from qorl.model.client import AstraModelClient, ModelClient
from qorl.model.schemas import ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.sft.dataset import validate_conversation, write_records
from qorl.sft.schemas import (
    Conversation,
    ConversationRequest,
    GenerationAttempt,
    GenerationIdentity,
    GenerationReport,
    GenerationSplitReport,
    PlanOnlyEvidence,
    PreparedDatasetManifest,
    PreparedDatasetSplit,
)
from qorl.taskset.schemas import Task, TaskRole, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json
from qorl.util.seeds import derive_seed
from qorl.util.time import utc_now
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool, start_pool
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest


def conversation_from_attempt(
    attempt: GenerationAttempt, *, source: Path, identity_sha256: str
) -> Conversation:
    """Use recorded typed requests, never current tools or provider-history reconstruction."""
    trace = attempt.trace
    if trace is None or not trace.model_responses:
        raise ValueError("no assistant response was recorded")
    if len(trace.model_requests) < len(trace.model_responses):
        raise ValueError("missing recorded model requests")
    requests: list[ConversationRequest] = []
    for request, response in zip(
        trace.model_requests, trace.model_responses, strict=False
    ):
        index = len(request.messages)
        if (
            trace.transcript[:index] != request.messages
            or trace.transcript[index] != response.message
        ):
            raise ValueError("recorded request does not match the preceding transcript")
        requests.append(
            ConversationRequest(assistant_message_index=index, tools=request.tools)
        )
    conversation = Conversation(
        conversation_id=attempt.attempt_id,
        task_id=attempt.task_id,
        messages=trace.transcript,
        requests=requests,
        metadata={
            "generation_attempt": (source / "attempts" / f"{attempt.attempt_id}.json")
            .resolve()
            .as_uri(),
            "generation_identity": (source / "identity.json").resolve().as_uri(),
            "generation_identity_sha256": identity_sha256,
            "generation_status": attempt.status.value,
            "plan_only": attempt.plan_only is not None,
            "agent_interface_version": trace.agent_interface_version,
        },
    )
    validate_conversation(conversation)
    return conversation


def generate_attempt(
    attempt: GenerationAttempt,
    task: Task,
    task_set: TaskSet,
    *,
    identity: GenerationIdentity,
    model: ModelSettings,
    client: ModelClient,
    pool: ContainerPool,
    slots: BoundedSemaphore,
    stop: Event,
    path: Path,
) -> GenerationAttempt:
    policy = QoAgentPolicy(
        client,
        identity.agent,
        context_length=model.context_length,
        max_tokens=identity.generation.inference.max_tokens,
        seed=attempt.model_seed,
    )
    evaluator: PlanValidationEvaluator[PostgresClient] | None = None
    worker: WorkerManifest | None = None
    error: BaseException | None = None
    write_json(path, attempt.model_dump(mode="json"))
    try:
        with slots:
            if stop.is_set():
                raise CancelledError("generation stopped before this attempt started")
            with pool.claim_worker() as slot:
                worker = slot.resources.manifest()
                if identity.generation.plan_only:
                    evaluator = PlanValidationEvaluator(
                        slot.client,
                        task_set,
                        task,
                        default_timeout_ms=seconds_to_ms(
                            identity.measurement.default_timeout_seconds
                        ),
                        max_candidates=identity.agent.candidate_attempts,
                        cancel=stop,
                    )
                else:
                    evaluator = RolloutEvaluator(
                        slot.client,
                        task_set,
                        task,
                        measurement=identity.measurement,
                        max_candidates=identity.agent.candidate_attempts,
                        cancel=stop,
                    )
                evaluator.start()
                trace = policy.search(
                    evaluator,
                    log_label=f"{task.task_id} attempt={attempt.attempt_id}",
                )
                if isinstance(evaluator, RolloutEvaluator):
                    evaluator.finish(
                        random.Random(attempt.measurement_seed),
                        selected_candidate_id=trace.selection.selected_candidate_id,
                    )
    except BaseException as caught:
        error = caught
    result = attempt.model_copy(
        update={
            "status": RunStatus.FAILED if error else RunStatus.COMPLETED,
            "completed_at_utc": utc_now(),
            "worker": worker,
            "trace": policy.trace,
            "rollout": evaluator.record(error)
            if isinstance(evaluator, RolloutEvaluator)
            else None,
            "plan_only": PlanOnlyEvidence(
                default=evaluator.default,
                candidates=evaluator.candidates,
                kept_default=evaluator.kept_default,
                selection=evaluator.selection,
            )
            if evaluator is not None and not isinstance(evaluator, RolloutEvaluator)
            else None,
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
        }
    )
    write_json(path, result.model_dump(mode="json"))
    if error is not None and not isinstance(error, Exception):
        raise error
    return result


def save_artifact(
    identity: GenerationIdentity,
    attempts: list[GenerationAttempt],
    output: Path,
    worker_pool: PoolManifest | None,
) -> Path:
    artifact = output / "conversations"
    splits: dict[TaskRole, PreparedDatasetSplit] = {}
    reports: dict[TaskRole, GenerationSplitReport] = {}
    for role, name in (
        (TaskRole.TRAIN, "training"),
        (TaskRole.VALIDATION, "validation"),
    ):
        selected = [attempt for attempt in attempts if attempt.split == role]
        conversations: list[Conversation] = []
        excluded: dict[str, str] = {}
        for attempt in selected:
            try:
                conversations.append(
                    conversation_from_attempt(
                        attempt,
                        source=output,
                        identity_sha256=sha256_file(output / "identity.json"),
                    )
                )
            except (ValueError, IndexError) as error:
                excluded[attempt.attempt_id] = str(error)
        write_records(artifact / f"{name}.jsonl", conversations)
        source = identity.selection_inputs[role]
        splits[role] = PreparedDatasetSplit(
            selection=identity.selections[role],
            selection_seed=source.selection_seed,
            generation_seed=identity.seed,
            selection_expression=source.expression,
            conversations=Path(f"{name}.jsonl"),
        )
        reports[role] = GenerationSplitReport(
            requested_attempts=len(selected),
            completed_attempts=sum(
                item.status == RunStatus.COMPLETED for item in selected
            ),
            failed_attempts=sum(item.status == RunStatus.FAILED for item in selected),
            saved_conversations=len(conversations),
            conversion_exclusions=excluded,
        )
    manifest = PreparedDatasetManifest(
        schema_version=2,
        format="qorl-conversations",
        training=splits[TaskRole.TRAIN],
        validation=splits[TaskRole.VALIDATION],
    )
    write_json(artifact / "manifest.json", manifest.model_dump(mode="json"))
    report = GenerationReport(
        training=reports[TaskRole.TRAIN],
        validation=reports[TaskRole.VALIDATION],
        worker_pool=worker_pool,
        files={
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name != "report.json"
            and not path.name.startswith(".")
        },
    )
    write_json(output / "report.json", report.model_dump(mode="json"))
    return artifact


def generate_dataset(
    config: SftExperimentConfig,
    selections: dict[TaskRole, TaskSelection],
    output: Path,
) -> Path:
    generation = config.data.generation
    if (
        generation is None
        or not isinstance(generation.model, ModelSettings)
        or not isinstance(generation.generations_per_task, int)
    ):
        raise ValueError(
            "configure data.generation.model and generations_per_task before preparation"
        )
    model = generation.model
    client = AstraModelClient(model, generation.inference)
    postgres = PostgresConfig.load(config.postgres.path)
    pool_config = load_pool_config(config.pool.path)
    selected = {
        role: selections[role] for role in (TaskRole.TRAIN, TaskRole.VALIDATION)
    }
    catalogs = {
        role: TaskSet.load(REPOSITORY_ROOT, selection.benchmark_id.value)
        for role, selection in selected.items()
    }
    identity_path = output / "identity.json"
    saved_identity = (
        GenerationIdentity.model_validate_json(identity_path.read_bytes())
        if output.exists()
        else None
    )
    identity = GenerationIdentity(
        run_id=saved_identity.run_id if saved_identity is not None else uuid4(),
        generation=generation,
        agent=config.agent,
        measurement=config.measurement,
        seed=config.experiment.seed,
        selections=selected,
        selection_inputs={
            TaskRole.TRAIN: config.data.training,
            TaskRole.VALIDATION: config.data.validation,
        },
        tasks={
            role: catalogs[role].resolve(selection)
            for role, selection in selected.items()
        },
        expected_pool=PoolManifest(
            id=pool_config.profile_id,
            path=str(pool_config.path),
            config_sha256=pool_config.sha256,
            worker_count=len(pool_config.workers),
            workers=[worker.manifest() for worker in pool_config.workers],
            postgres_config=postgres.manifest(),
        ),
        system_prompt=system_prompt(config.agent.candidate_attempts),
    )
    if saved_identity is not None:
        if saved_identity != identity:
            raise ValueError(
                "generation inputs changed; start a new run or reuse the explicit conversation artifact"
            )
    else:
        output.mkdir(parents=True)
        write_json(identity_path, identity.model_dump(mode="json"))
    report_path = output / "report.json"
    if report_path.exists():
        report = GenerationReport.model_validate_json(report_path.read_bytes())
        for name, checksum in report.files.items():
            if sha256_file(output / name) != checksum:
                raise ValueError(f"saved generation evidence changed: {name}")
        return output / "conversations"
    attempts: list[GenerationAttempt] = []
    pending: list[tuple[GenerationAttempt, Task, TaskSet]] = []
    for role, tasks in identity.tasks.items():
        for task in tasks:
            for index in range(generation.generations_per_task):
                identifier = sha256_json(
                    [identity.run_id.hex, role.value, task.task_id, index]
                )[:24]
                path = output / "attempts" / f"{identifier}.json"
                if path.exists():
                    attempt = GenerationAttempt.model_validate_json(path.read_bytes())
                    if attempt.status == RunStatus.RUNNING:
                        attempt = attempt.model_copy(
                            update={
                                "status": RunStatus.FAILED,
                                "completed_at_utc": utc_now(),
                                "error_type": "InterruptedAttempt",
                                "error": "previous attempt did not finish; no replacement generation",
                            }
                        )
                        write_json(path, attempt.model_dump(mode="json"))
                    attempts.append(attempt)
                else:
                    attempt = GenerationAttempt(
                        attempt_id=identifier,
                        split=role,
                        task_id=task.task_id,
                        generation_index=index,
                        model_seed=derive_seed(
                            identity.seed, "generation-model", task.task_id, str(index)
                        ),
                        measurement_seed=derive_seed(
                            identity.seed, "generation-pairs", task.task_id, str(index)
                        ),
                        status=RunStatus.RUNNING,
                        started_at_utc=utc_now(),
                    )
                    pending.append((attempt, task, catalogs[role]))
    stop = Event()
    pool: ContainerPool | None = None
    failure: BaseException | None = None
    try:
        if pending:
            pool = start_pool(
                f"qorl-generate-{uuid4().hex}",
                REPOSITORY_ROOT / "data/imdb.tar.gz",
                postgres_config=postgres,
                pool_config=pool_config,
            )
            for slot in pool.workers:
                capture_environment(
                    pool, slot, output / f"worker-{slot.resources.index}", "pre"
                )
            slots = BoundedSemaphore(
                min(len(pool.workers), model.max_concurrent_requests)
            )
            with ThreadPoolExecutor(max_workers=len(pool.workers)) as executor:
                futures = [
                    executor.submit(
                        generate_attempt,
                        attempt,
                        task,
                        task_set,
                        identity=identity,
                        model=model,
                        client=client,
                        pool=pool,
                        slots=slots,
                        stop=stop,
                        path=output / "attempts" / f"{attempt.attempt_id}.json",
                    )
                    for attempt, task, task_set in pending
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()
                    while True:
                        try:
                            wait(futures)
                            break
                        except KeyboardInterrupt:
                            continue
                    raise
    except BaseException as error:
        failure = error
    finally:
        if pool is not None:
            pool.close()
    for attempt, _, _ in pending:
        path = output / "attempts" / f"{attempt.attempt_id}.json"
        if path.exists():
            attempt = GenerationAttempt.model_validate_json(path.read_bytes())
        if attempt.status == RunStatus.RUNNING:
            attempt = attempt.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "completed_at_utc": utc_now(),
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                }
            )
            write_json(path, attempt.model_dump(mode="json"))
        attempts.append(attempt)
    artifact = save_artifact(
        identity,
        sorted(
            attempts,
            key=lambda item: (item.split.value, item.task_id, item.generation_index),
        ),
        output,
        pool.manifest() if pool is not None else None,
    )
    if failure is not None:
        raise failure
    return artifact
