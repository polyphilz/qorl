"""Summarize native rollout evidence and credit annotations without re-scoring it."""

from collections import Counter
from pathlib import Path

from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk

from qorl.agent.agent import total_usage
from qorl.evaluation.evaluate import summarize_performance
from qorl.measure.schemas import OutcomeKind, RolloutRecord
from qorl.model.schemas import TokenUsage
from qorl.rl.schemas import (
    AnchoredCredit,
    Annotation,
    EpisodeEvidence,
    EpisodeFailure,
    LearningEvidence,
    RlTrainingReport,
    UpdateMetric,
)
from qorl.util.io import write_json


def write_report(
    training: Path, *, completed: bool, anchored: bool
) -> RlTrainingReport:
    """Read the native append-only records, retaining raw speedups and actual credit."""
    credits: dict[str, AnchoredCredit] = {}
    ships: dict[str, int] = {}
    advantages: dict[str, float] = {}
    for producer in sorted(get_annotations_dir(training).glob("*")):
        if not producer.is_dir():
            continue
        for chunk in sorted(chunk_numbers(producer)):
            with open_chunk(producer, chunk) as stream:
                for line in stream:
                    annotation = Annotation.model_validate_json(line)
                    if annotation.info.qorl_advantage is not None:
                        credits[annotation.trace_id] = annotation.info.qorl_advantage
                    if annotation.info.ship is not None:
                        ships[annotation.trace_id] = annotation.info.ship.step
                    values = {
                        value
                        for branch in annotation.branches
                        for value in branch.advantages or []
                        if value != 0
                    }
                    if len(values) > 1:
                        raise ValueError(
                            "QORL expects one scalar advantage per conversation"
                        )
                    if any(
                        branch.advantages is not None for branch in annotation.branches
                    ):
                        advantages[annotation.trace_id] = next(iter(values), 0.0)
    episode_count = 0
    failures: list[EpisodeFailure] = []
    records: list[RolloutRecord] = []
    learning: list[LearningEvidence] = []
    usage: list[TokenUsage] = []
    missing_usage = 0
    directory = get_trace_stream(training)
    for chunk in sorted(chunk_numbers(directory)):
        with open_chunk(directory, chunk) as stream:
            for line in stream:
                episode = EpisodeEvidence.model_validate_json(line)
                episode_count += 1
                if not episode.ok:
                    failures.append(
                        EpisodeFailure(
                            episode_id=episode.id,
                            group_id=episode.group.id,
                            task_id=episode.task.data.task_id,
                            errors=episode.errors,
                        )
                    )
                policy = episode.run.work.policy
                for trace in episode.traces:
                    if trace.info.qorl_policy is not None:
                        usage.append(trace.info.qorl_policy.usage)
                    else:
                        missing_usage += 1
                    record = trace.info.qorl
                    if record is not None:
                        records.append(record)
                    learning.append(
                        LearningEvidence(
                            episode_id=episode.id,
                            group_id=episode.group.id,
                            trace_id=trace.id,
                            task_id=record.task_id if record is not None else None,
                            policy_start=policy.start if policy is not None else None,
                            policy_end=policy.end if policy is not None else None,
                            ship_step=ships.get(trace.id),
                            anchored=credits.get(trace.id, trace.info.qorl_advantage),
                            assigned_advantage=advantages.get(trace.id),
                        )
                    )
    updates: set[int] = set()
    metrics = training / "monitors/file/metrics.jsonl"
    if metrics.is_file():
        with metrics.open() as stream:
            for line in stream:
                metric = UpdateMetric.model_validate_json(line)
                if (
                    metric.producer == "trainer"
                    and metric.learning_rate is not None
                    and metric.step is not None
                ):
                    updates.add(metric.step)
    counts = Counter(
        record.final.kind for record in records if record.final is not None
    )
    report = RlTrainingReport(
        completed=completed,
        scalar_reward_hook="unused for anchored credit (0.0)"
        if anchored
        else "ordinary GRPO scalar reward",
        optimizer_steps=sorted(updates),
        episode_count=episode_count,
        episode_failure_count=len(failures),
        episode_failures=failures,
        outcome_counts={kind: counts[kind] for kind in OutcomeKind},
        outcome_rates={
            kind: counts[kind] / sum(counts.values()) if counts else None
            for kind in OutcomeKind
        },
        performance=summarize_performance(records),
        usage=total_usage(usage),
        policy_usage_missing_count=missing_usage,
        learning=learning,
        checkpoints=sorted(
            path.parent
            for path in (training / "checkpoints").glob("step_*/trainer/.metadata")
        ),
    )
    write_json(training / "report.json", report.model_dump(mode="json"))
    return report
