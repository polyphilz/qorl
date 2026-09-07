import math

import pytest
from prime_rl.orchestrator.algo.qorl_anchored_grpo import (
    QorlDecision,
    RolloutScope,
    anchored_advantages,
    decision_from_final,
    share_reusable_speedups,
)

from qorl.evaluation.benchmark import summarize
from qorl.measure.schemas import (
    MeasuredOutcome,
    NoValidCandidateOutcome,
    PairedMeasurements,
    RolloutFailure,
    RolloutRecord,
    TimedOutOutcome,
)
from qorl.rl.reward import scalar_reward, training_speedup
from qorl.rl.schemas import RlRolloutRecord, ScalarRewardSettings

SETTINGS = ScalarRewardSettings(
    invalid_attempt_penalty=0.1,
    duplicate_attempt_penalty=0.05,
    timeout_attempt_penalty=0.2,
    no_valid_candidate_reward=-1.0,
)


def test_training_clips_without_changing_observations_or_reports(
    rollout_record: RolloutRecord,
) -> None:
    outcome = rollout_record.final
    assert isinstance(outcome, MeasuredOutcome)
    pairs = outcome.paired.model_dump()
    for pair in pairs["measurements"]:
        pair["candidate"]["execution_time_ms"] = 0.5
    final = MeasuredOutcome.model_validate(
        {
            **outcome.model_dump(),
            "paired": pairs,
            "candidate_median_execution_time_ms": 0.5,
            "speedup": 20.0,
        }
    )
    record = RolloutRecord.model_validate(
        {**rollout_record.model_dump(), "final": final}
    )
    assert training_speedup(record) == 10.0
    assert scalar_reward(record, SETTINGS) == pytest.approx(math.log(10.0))
    assert record.final is not None and record.final.speedup == 20.0
    summary = summarize([{"rollout": record.to_wire()}])
    assert summary["geometric_mean_speedup"] == pytest.approx(20.0)
    assert summary["total_workload_speedup"] == 20.0


def test_timeout_proxy_is_training_only(rollout_record: RolloutRecord) -> None:
    candidate = rollout_record.candidates[0].model_copy(
        update={"execution_timed_out": True, "timeout_ms": 5000}
    )
    final = TimedOutOutcome(
        selected_candidate_id=candidate.candidate_id,
        selected_plan_sha256=candidate.plan_sha256,
        timing_reuse_key=candidate.timing_reuse_key,
        initial_default_median_execution_time_ms=10.0,
        timeout_ms=5000,
        paired=PairedMeasurements(),
    )
    record = RolloutRecord.model_validate(
        {**rollout_record.model_dump(), "candidates": [candidate], "final": final}
    )
    assert training_speedup(record) == 0.1
    assert scalar_reward(record, SETTINGS) == pytest.approx(
        math.log(0.1) - SETTINGS.timeout_attempt_penalty
    )
    assert record.final is not None and record.final.speedup is None
    summary = summarize([{"rollout": record.to_wire()}])
    assert summary["geometric_mean_speedup"] is None
    assert summary["timeout_count"] == 1
    assert summary["failure_count"] == 0


def test_no_valid_decision_and_unscored_failure_are_different(
    rollout_record: RolloutRecord,
) -> None:
    invalid = RolloutRecord.model_validate(
        {
            **rollout_record.model_dump(),
            "candidates": [],
            "final": NoValidCandidateOutcome(),
        }
    )
    assert training_speedup(invalid) is None
    assert scalar_reward(invalid, SETTINGS) == -1.0
    failed = RolloutRecord.model_validate(
        {
            **rollout_record.model_dump(),
            "final": None,
            "failure": RolloutFailure(
                operation="initial_default",
                error_type="QueryTimeout",
                error="timeout",
                paired=PairedMeasurements(),
            ),
        }
    )
    assert training_speedup(failed) is None
    with pytest.raises(ValueError, match="unscored"):
        scalar_reward(failed, SETTINGS)


def test_serialized_outcome_and_pool_identity_reach_the_pinned_fork(
    rl_rollout_record: RlRolloutRecord,
) -> None:
    selected = rl_rollout_record.candidates[0]
    earlier = selected.model_copy(
        update={
            "candidate_id": "earlier-timeout",
            "constraints_satisfied": False,
            "execution_timed_out": True,
            "timeout_ms": 5000,
            "plan_sha256": None,
            "timing_reuse_key": None,
        }
    )
    record = RlRolloutRecord.model_validate(
        {**rl_rollout_record.model_dump(), "candidates": [earlier, selected]}
    )
    scope = RolloutScope.model_validate(record.to_wire())
    assert scope.task_id == record.task_id
    assert scope.database_pool.config_sha256 == record.database_pool.config_sha256
    assert (
        scope.database_pool.postgres_config.pg_conf_sha256
        == record.database_pool.postgres_config.pg_conf_sha256
    )
    assert record.final is not None
    decision = decision_from_final(record.final.to_wire())
    assert decision.kind == "candidate"
    assert decision.observed_speedup == 2.0
    assert decision.timing_reuse_key == selected.timing_reuse_key
    siblings = share_reusable_speedups(
        [
            decision,
            QorlDecision("keep_default"),
            QorlDecision("keep_default"),
            QorlDecision("keep_default"),
        ]
    )
    advantages = anchored_advantages(
        siblings, tau=math.log(1.05), c=1.0, d=0.05, t=0.2, min_peers=2
    )
    assert advantages[0].advantage == pytest.approx(math.log(2.0) - math.log(1.05))
    assert advantages[0].protocol_cost == 0.0
    assert advantages[0].kind == "candidate"
