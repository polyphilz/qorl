"""Summarize native rollout evidence and credit annotations without re-scoring it."""

from collections import Counter
from pathlib import Path

from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1.episode import GroupInfo, TrainRunInfo
from verifiers.v1.trace import Error, TraceTask

from qorl.evaluation.evaluate import summarize_performance
from qorl.evaluation.schemas import PerformanceSummary
from qorl.measure.schemas import OutcomeKind, RolloutRecord
from qorl.rl.schemas import RlRolloutRecord
from qorl.rl.tasks import QorlTaskData
from qorl.util.io import write_json


class AnchoredCredit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    discarded: bool
    discard_reason: str | None = None
    quality: float | None = None
    reference: float | None = None
    protocol_cost: float
    advantage: float


class ShipInfo(BaseModel):
    step: int


class TraceInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    qorl: RlRolloutRecord | None = None
    qorl_advantage: AnchoredCredit | None = None
    ship: ShipInfo | None = None


class TraceEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    info: TraceInfo


class EpisodeEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    group: GroupInfo
    run: TrainRunInfo
    traces: list[TraceEvidence]
    task: TraceTask[QorlTaskData]
    ok: bool
    errors: list[Error] = []


class BranchCredit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    advantages: list[float] | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    trace_id: str
    info: TraceInfo
    branches: list[BranchCredit] = []


class LearningEvidence(BaseModel):
    """Join keys into native episodes, ship annotations, metrics and checkpoints."""

    episode_id: str
    group_id: str
    trace_id: str
    task_id: str | None
    policy_start: int | None
    policy_end: int | None
    ship_step: int | None
    anchored: AnchoredCredit | None
    assigned_advantage: float | None


class EpisodeFailure(BaseModel):
    episode_id: str
    group_id: str
    task_id: str
    errors: list[Error]


class UpdateMetric(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    producer: str | None = None
    step: int | None = None
    learning_rate: float | None = Field(default=None, alias="optim/lr")
    gradient_norm: float | None = Field(default=None, alias="optim/grad_norm")


class RlTrainingReport(BaseModel):
    schema_version: int = 1
    completed: bool
    scalar_reward_hook: str
    optimizer_steps: list[int]
    episode_count: int
    episode_failure_count: int
    episode_failures: list[EpisodeFailure]
    outcome_counts: dict[OutcomeKind, int]
    outcome_rates: dict[OutcomeKind, float | None]
    performance: PerformanceSummary
    learning: list[LearningEvidence]
    checkpoints: list[Path]


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
                    record = trace.info.qorl
                    if (
                        record is not None
                        and record.final is not None
                        and record.failure is None
                    ):
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
            kind: counts[kind] / len(records) if records else None
            for kind in OutcomeKind
        },
        performance=summarize_performance(records),
        learning=learning,
        checkpoints=sorted(
            path.parent
            for path in (training / "checkpoints").glob("step_*/trainer/.metadata")
        ),
    )
    write_json(training / "report.json", report.model_dump(mode="json"))
    return report
