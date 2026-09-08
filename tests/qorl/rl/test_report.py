"""Join native arrival, group-credit and optimizer evidence without re-scoring."""

import json
import random
from pathlib import Path

import pytest
from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from tests.qorl.measure.test_feedback import ENABLED
from tests.qorl.measure.test_rollout import ACTION, Worker, evaluator
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.episode import (
    Episode,
    GroupInfo,
    PolicySpan,
    TrainRunInfo,
    TrainWorkInfo,
)
from verifiers.v1.state import State
from verifiers.v1.trace import Error, TraceTask

from qorl.measure.schemas import OutcomeKind
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import ExplainResult
from qorl.rl.report import (
    AnchoredCredit,
    Annotation,
    BranchCredit,
    EpisodeEvidence,
    ShipInfo,
    TraceEvidence,
    TraceInfo,
    write_report,
)
from qorl.rl.schemas import RlRolloutRecord
from qorl.rl.tasks import QorlTaskData


def test_report_keeps_credit_for_unshipped_groups_and_raw_speedups(
    tmp_path: Path, rl_rollout_record: RlRolloutRecord
) -> None:
    stream = get_trace_stream(tmp_path)
    stream.mkdir(parents=True)
    episode = EpisodeEvidence(
        id="episode",
        ok=True,
        task=TraceTask(
            type="QorlTask", data=QorlTaskData(task_id="job/1a", template_id="1")
        ),
        group=GroupInfo(id="query-group"),
        run=TrainRunInfo(
            id="native-run",
            work=TrainWorkInfo(step=2, policy=PolicySpan(start=1, end=1)),
        ),
        traces=[TraceEvidence(id="trace", info=TraceInfo(qorl=rl_rollout_record))],
    )
    failed = Episode[QorlTaskData, State, AgentConfig](
        id="failed",
        task=episode.task,
        group=episode.group,
        run=episode.run,
        ok=False,
        errors=[Error(type="ConnectionError", message="database unavailable")],
    )
    (stream / "00000.jsonl").write_text(
        episode.model_dump_json() + "\n" + json.dumps(failed.to_record()) + "\n"
    )
    discarded = episode.model_copy(
        update={
            "id": "discarded",
            "group": GroupInfo(id="discarded-group"),
            "traces": [TraceEvidence(id="discarded-trace", info=TraceInfo())],
        }
    )
    with (stream / "00000.jsonl").open("a") as output:
        output.write(discarded.model_dump_json() + "\n")
    annotations = get_annotations_dir(tmp_path) / "orchestrator"
    annotations.mkdir(parents=True)
    credit = AnchoredCredit(
        discarded=False,
        quality=0.643,
        reference=0.0,
        protocol_cost=0.0,
        advantage=0.643,
    )
    (annotations / "00000.jsonl").write_text(
        Annotation(
            trace_id="trace", info=TraceInfo(qorl_advantage=credit)
        ).model_dump_json()
        + "\n"
    )
    discard_credit = AnchoredCredit(
        discarded=True,
        discard_reason="incomplete_group",
        protocol_cost=0.0,
        advantage=0.0,
    )
    with (annotations / "00000.jsonl").open("a") as output:
        output.write(
            Annotation(
                trace_id="discarded-trace",
                info=TraceInfo(qorl_advantage=discard_credit),
            ).model_dump_json()
            + "\n"
        )
    # Group-finalization credit exists even before this conversation ships.
    partial = write_report(tmp_path, completed=False, anchored=True)
    assert partial.learning[0].anchored == credit
    assert partial.learning[0].ship_step is None
    assert partial.learning[0].assigned_advantage is None
    assert partial.performance.geometric_mean_speedup == 2.0
    with (annotations / "00000.jsonl").open("a") as output:
        output.write(
            Annotation(
                trace_id="trace",
                info=TraceInfo(ship=ShipInfo(step=2)),
                branches=[BranchCredit(advantages=[0, 0.643, 0.643])],
            ).model_dump_json()
            + "\n"
        )
    (tmp_path / "monitors/file/metrics.jsonl").write_text(
        '{"producer":"trainer","step":2,"optim/lr":0.000001,"optim/grad_norm":0.2}\n'
    )
    final = write_report(tmp_path, completed=True, anchored=True)
    assert final.episode_count == 3
    assert final.episode_failure_count == 1
    assert final.episode_failures[0].task_id == "job/1a"
    assert final.episode_failures[0].errors[0].message == "database unavailable"
    assert final.learning[1].anchored == discard_credit
    assert final.learning[1].ship_step is None
    assert sum(final.outcome_counts.values()) == 1
    assert sum(rate for rate in final.outcome_rates.values() if rate is not None) == 1.0
    row = final.learning[0]
    assert (
        row.episode_id,
        row.group_id,
        row.trace_id,
        row.policy_start,
        row.policy_end,
        row.ship_step,
    ) == ("episode", "query-group", "trace", 1, 1, 2)
    assert row.assigned_advantage == 0.643
    assert final.optimizer_steps == [2]
    assert final.scalar_reward_hook == "unused for anchored credit (0.0)"


def test_report_counts_probe_work_on_unscored_failure(
    tmp_path: Path, rl_rollout_record: RlRolloutRecord
) -> None:
    class FailedProbeWorker(Worker):
        def explain(
            self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
        ) -> ExplainResult:
            if (
                analyze
                and hint
                and any(call.analyze and call.candidate for call in self.calls)
            ):
                raise PostgresError("connection lost after feedback warmup")
            return super().explain(sql, timeout_ms, analyze=analyze, hint=hint)

    run = evaluator(FailedProbeWorker(), settings=ENABLED)
    run.start()
    with pytest.raises(PostgresError) as caught:
        run.evaluate(ACTION)
    failed_record = RlRolloutRecord.model_validate(
        rl_rollout_record.model_dump()
        | run.record(caught.value).model_dump()
        | {"scalar_reward": None}
    )
    selection_run = evaluator(Worker())
    selection_run.start()
    selection_run.evaluate(ACTION)
    selection_run.reject_selection(
        {"selected_candidate_id": "invented"}, ["not an eligible issued candidate"]
    )
    selection_run.resolve_selection()
    selection_run.finish(random.Random(0))
    selection_record = RlRolloutRecord.model_validate(
        rl_rollout_record.model_dump()
        | selection_run.record().model_dump()
        | {"scalar_reward": None}
    )
    stream = get_trace_stream(tmp_path)
    stream.mkdir(parents=True)
    episodes: list[EpisodeEvidence] = []
    for index, record in enumerate(
        (rl_rollout_record, failed_record, selection_record)
    ):
        episodes.append(
            EpisodeEvidence(
                id=f"episode-{index}",
                ok=record.failure is None,
                task=TraceTask(
                    type="QorlTask",
                    data=QorlTaskData(
                        task_id=record.task_id, template_id=record.template_id
                    ),
                ),
                group=GroupInfo(id=f"group-{index}"),
                run=TrainRunInfo(id="native-run", work=TrainWorkInfo(step=1)),
                traces=[
                    TraceEvidence(id=f"trace-{index}", info=TraceInfo(qorl=record))
                ],
                errors=[]
                if record.failure is None
                else [Error(type="PostgresError", message=record.failure.error)],
            )
        )
    (stream / "00000.jsonl").write_text(
        "\n".join(episode.model_dump_json() for episode in episodes) + "\n"
    )
    result = write_report(tmp_path, completed=False, anchored=True)
    assert result.episode_failure_count == 1
    assert result.performance.failure_count == 1
    assert result.performance.selection_failure_count == 1
    assert result.outcome_counts[OutcomeKind.SELECTION_FAILED] == 1
    assert result.performance.execution_counts.candidate_feedback == 2
    assert (
        sum(rate for rate in result.outcome_rates.values() if rate is not None) == 1.0
    )
    assert result.performance.geometric_mean_speedup == 2.0
    feedback = failed_record.candidates[0].execution_feedback
    assert feedback is not None and len(feedback.warmups) == 1
    assert feedback.warmups[0].analyzed_document is not None
