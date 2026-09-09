"""Run independent agent conversations and report their unmodified plan evidence."""

import math
import os
import random
import time
from collections import Counter
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from threading import Event
from uuid import uuid4

from qorl.agent.agent import QoAgentPolicy, total_usage
from qorl.agent.schemas import AgentSettings
from qorl.agent.types import StopReason
from qorl.evaluation.schemas import (
    EvaluationItem,
    EvaluationReport,
    EvaluationRollout,
    EvaluationSettings,
    EvaluationSummary,
    PerformanceSummary,
)
from qorl.inference.local import serve_local_model
from qorl.measure.environment import capture_environment
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import (
    DefaultDuplicateOutcome,
    ExecutionCounts,
    KeptDefaultOutcome,
    MeasuredOutcome,
    OutcomeKind,
    PairedMeasurements,
    RolloutFailure,
    RolloutMeasurementSettings,
    RolloutRecord,
    RunStatus,
)
from qorl.model.client import ModelClient, model_client
from qorl.model.exceptions import ModelRequestError
from qorl.model.schemas import InferenceSettings, LocalInferenceSettings, ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.taskset.schemas import TaskRole, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.io import write_json
from qorl.util.seeds import derive_seed
from qorl.util.time import utc_now
from qorl.worker_pool.containers import ContainerPool, start_pool
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import PoolConfig, WorkerManifest

ROLLOUT_NUMBER_WIDTH = 3


def summarize_performance(records: list[RolloutRecord]) -> PerformanceSummary:
    """Use only known timings, reporting the exclusions alongside the raw speedups."""
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
    counts = [
        record.execution_counts
        for record in records
        if record.execution_counts is not None
    ]
    selected_positions: Counter[int] = Counter()
    earlier_selections = 0
    for record in records:
        if record.final is None or record.final.selected_candidate_id is None:
            continue
        for position, candidate in enumerate(record.candidates, start=1):
            if candidate.candidate_id == record.final.selected_candidate_id:
                selected_positions[position] += 1
                earlier_selections += position < len(record.candidates)
    return PerformanceSummary(
        rollout_count=len(records),
        scored_rollout_count=len(speedups),
        failure_count=len(records) - len(outcomes),
        timeout_count=sum(
            outcome.kind == OutcomeKind.TIMED_OUT for outcome in outcomes
        ),
        no_valid_candidate_count=sum(
            outcome.kind == OutcomeKind.NO_VALID_CANDIDATE for outcome in outcomes
        ),
        selection_failure_count=sum(
            outcome.kind == OutcomeKind.SELECTION_FAILED for outcome in outcomes
        ),
        selected_candidate_positions=dict(sorted(selected_positions.items())),
        earlier_candidate_selection_count=earlier_selections,
        rejected_selection_count=sum(
            len(record.selection.rejections)
            for record in records
            if record.selection is not None
        ),
        geometric_mean_speedup=math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        )
        if speedups
        else None,
        candidate_workload_time_ms=candidate_time,
        default_workload_time_ms=default_time,
        total_workload_speedup=default_time / candidate_time
        if candidate_time
        else None,
        regression_count=sum(value < 1 for value in speedups),
        execution_counts=ExecutionCounts(
            initial_default=sum(item.initial_default for item in counts),
            candidate_feedback=sum(item.candidate_feedback for item in counts),
            final_paired=sum(item.final_paired for item in counts),
        ),
        execution_accounting_missing=len(records) - len(counts),
    )


def summarize_evaluation(
    records: list[EvaluationRollout], expected_rollouts: int
) -> EvaluationSummary:
    """Count saved evidence, keeping structural novelty independent of timing reuse."""
    rollouts = [record.rollout for record in records]
    valid = sum(
        any(
            candidate.action_valid and candidate.constraints_satisfied
            for candidate in rollout.candidates
        )
        for rollout in rollouts
    )
    novel = sum(
        any(candidate.structurally_novel for candidate in rollout.candidates)
        for rollout in rollouts
    )
    distinct = {
        (rollout.task_id, candidate.structural_plan_sha256)
        for rollout in rollouts
        for candidate in rollout.candidates
        if candidate.structurally_novel
    }
    count = len(records)
    return EvaluationSummary(
        recorded_rollout_count=count,
        unrecorded_rollout_count=expected_rollouts - count,
        recorded_task_count=len({rollout.task_id for rollout in rollouts}),
        valid_plan_rollout_count=valid,
        novel_plan_rollout_count=novel,
        distinct_novel_plan_count=len(distinct),
        valid_plan_rate=valid / count if count else None,
        novel_plan_rate=novel / count if count else None,
        distinct_novel_plan_yield=len(distinct) / count if count else None,
        candidate_attempt_count=sum(len(rollout.candidates) for rollout in rollouts),
        outcome_counts={
            kind: sum(
                rollout.final is not None and rollout.final.kind == kind
                for rollout in rollouts
            )
            for kind in OutcomeKind
        },
        stop_reason_counts={
            reason: sum(
                record.trace is not None and record.trace.stop_reason == reason
                for record in records
            )
            for reason in StopReason
        },
        usage=total_usage([record.trace.usage for record in records if record.trace]),
        performance=summarize_performance(rollouts),
    )


def evaluate_rollout(
    item: EvaluationItem,
    *,
    pool: ContainerPool,
    task_set: TaskSet,
    client: ModelClient,
    model: ModelSettings,
    inference: InferenceSettings,
    agent: AgentSettings,
    measurement: RolloutMeasurementSettings,
    seed: int,
    stop: Event,
    output_dir: Path,
) -> EvaluationRollout:
    """Save a completed or failed attempt before releasing it to the task scheduler."""
    if stop.is_set():
        raise CancelledError("evaluation stopped before this rollout started")
    started = utc_now()
    model_seed = derive_seed(
        seed, "evaluation-model", item.task.task_id, str(item.rollout_index)
    )
    measurement_seed = derive_seed(
        seed, "evaluation-pairs", item.task.task_id, str(item.rollout_index)
    )
    policy = QoAgentPolicy(
        client,
        agent,
        context_length=model.context_length,
        max_tokens=inference.max_tokens,
        seed=model_seed,
    )
    evaluator: RolloutEvaluator[PostgresClient] | None = None
    worker: WorkerManifest | None = None
    error: BaseException | None = None
    try:
        with pool.claim_worker() as slot:
            worker = slot.resources.manifest()
            evaluator = RolloutEvaluator(
                slot.client,
                task_set,
                item.task,
                measurement=measurement,
                max_candidates=agent.candidate_attempts,
                cancel=stop,
            )
            evaluator.start()
            policy_trace = policy.search(
                evaluator,
                log_label=f"{item.task.task_id} rollout={item.rollout_index}",
            )
            evaluator.finish(
                random.Random(measurement_seed),
                selected_candidate_id=policy_trace.selection.selected_candidate_id,
            )
    except BaseException as caught:
        error = caught
        if not isinstance(caught, (PostgresError, ContainerError)):
            stop.set()

    record = EvaluationRollout(
        rollout_index=item.rollout_index,
        model_seed=model_seed,
        measurement_seed=measurement_seed,
        started_at_utc=started,
        completed_at_utc=utc_now(),
        worker=worker,
        rollout=evaluator.record(error)
        if evaluator is not None
        else RolloutRecord(
            task_id=item.task.task_id,
            template_id=item.task.template_id,
            measurement=measurement,
            default=None,
            candidates=[],
            final=None,
            failure=RolloutFailure(
                operation="setup",
                error_type=type(error).__name__,
                error=str(error),
                paired=PairedMeasurements(),
            ),
        ),
        trace=policy.trace,
    )
    write_json(
        output_dir
        / "rollouts"
        / item.task.task_id
        / f"{item.rollout_index:0{ROLLOUT_NUMBER_WIDTH}d}.json",
        record.model_dump(mode="json"),
    )
    # Query/worker errors affect this rollout. Provider/setup/programming errors
    # stop the run, after retaining their evidence and draining other workers.
    if error is not None and not isinstance(error, (PostgresError, ContainerError)):
        raise error
    return record


def evaluate(
    task_set: TaskSet,
    selection: TaskSelection,
    output_dir: Path,
    *,
    split: TaskRole,
    model: ModelSettings,
    inference: InferenceSettings,
    serving_gpu_ids: list[int] | None,
    agent: AgentSettings,
    measurement: RolloutMeasurementSettings,
    settings: EvaluationSettings,
    seed: int,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> EvaluationReport:
    """Own serving and PostgreSQL until all threads finish; never resume or overwrite."""
    tasks = task_set.resolve(selection)
    expected = len(tasks) * settings.rollouts_per_task
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"evaluation output is not empty; refusing to overwrite: {output_dir}"
        )
    report_path = output_dir / "evaluation.json"
    # Exclusively claim the report even when the run dispatcher allocated the directory.
    with report_path.open("x"):
        pass
    report = EvaluationReport(
        status=RunStatus.RUNNING,
        split=split,
        selection=selection,
        model=model,
        seed=seed,
        settings=settings,
        plan_fingerprint_version=PLAN_FINGERPRINT_VERSION,
        started_at_utc=utc_now(),
        summary=summarize_evaluation([], expected),
    )
    write_json(report_path, report.model_dump(mode="json"))
    started = time.monotonic()
    stop = Event()
    try:
        with ExitStack() as resources:
            client: ModelClient
            if isinstance(inference, LocalInferenceSettings):
                local = resources.enter_context(
                    serve_local_model(
                        model,
                        inference,
                        serving_gpu_ids or [],
                        output_dir / "server.log",
                    )
                )
                client = local
                report.local_server = local.identity
            else:
                client = model_client(model, inference)
                if model.api_key_env is None or not os.environ.get(model.api_key_env):
                    raise ModelRequestError(
                        f"missing model credential: {model.api_key_env}"
                    )
            write_json(report_path, report.model_dump(mode="json"))
            pool = start_pool(
                f"qorl-eval-{uuid4().hex}",
                REPOSITORY_ROOT / "data/imdb.tar.gz",
                postgres_config=postgres_config,
                pool_config=pool_config,
            )
            resources.callback(pool.close)
            report.worker_pool = pool.manifest()
            write_json(report_path, report.model_dump(mode="json"))
            for slot in pool.workers:
                capture_environment(
                    pool, slot, output_dir / f"worker-{slot.resources.index}", "pre"
                )
            work = partial(
                evaluate_rollout,
                pool=pool,
                task_set=task_set,
                client=client,
                model=model,
                inference=inference,
                agent=agent,
                measurement=measurement,
                seed=seed,
                stop=stop,
                output_dir=output_dir,
            )
            with ThreadPoolExecutor(max_workers=len(pool.workers)) as executor:
                futures: list[Future[EvaluationRollout]] = []
                try:
                    for task in tasks:
                        for index in range(settings.rollouts_per_task):
                            futures.append(
                                executor.submit(work, EvaluationItem(task, index))
                            )
                    for ordinal, future in enumerate(as_completed(futures), start=1):
                        try:
                            result = future.result()
                        except CancelledError:
                            # A sibling's fatal error sets stop before its record
                            # is saved. Surface that error, not a queued cancellation.
                            if stop.is_set():
                                continue
                            raise
                        final = result.rollout.final
                        detail = final.kind.value if final is not None else "failed"
                        print(
                            f"[{ordinal}/{expected}] {result.rollout.task_id} "
                            f"rollout={result.rollout_index} {detail}",
                            flush=True,
                        )
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    # Finish in-flight calls before SQL/server teardown, even if
                    # another interrupt arrives during that wait.
                    while True:
                        try:
                            wait(futures)
                            break
                        except KeyboardInterrupt:
                            continue
                    raise
            for slot in pool.workers:
                capture_environment(
                    pool, slot, output_dir / f"worker-{slot.resources.index}", "post"
                )
    except BaseException as error:
        report.status = (
            RunStatus.INTERRUPTED
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else RunStatus.FAILED
        )
        report.error = str(error) or type(error).__name__
        raise
    else:
        report.status = RunStatus.COMPLETED
    finally:
        # Read the files after workers and resources have stopped. This includes
        # results saved while the main thread was handling an interruption.
        records = [
            EvaluationRollout.model_validate_json(path.read_bytes())
            for path in sorted((output_dir / "rollouts").glob("*/*.json"))
        ]
        report.summary = summarize_evaluation(records, expected)
        if (
            report.status == RunStatus.COMPLETED
            and report.summary.performance.failure_count
        ):
            report.status = RunStatus.COMPLETED_WITH_FAILURES
        report.completed_at_utc = utc_now()
        report.elapsed_seconds = time.monotonic() - started
        write_json(report_path, report.model_dump(mode="json"))
    if report.status == RunStatus.COMPLETED_WITH_FAILURES:
        raise RuntimeError(
            f"evaluation completed with {report.summary.performance.failure_count} failed rollouts; "
            f"results: {output_dir}"
        )
    return report
