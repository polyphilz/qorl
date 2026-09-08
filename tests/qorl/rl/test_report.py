"""Join native arrival, group-credit and optimizer evidence without re-scoring."""

import json
from pathlib import Path

from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
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
