import pytest

from qorl.evaluation.evaluate import summarize_performance
from qorl.measure.schemas import (
    NoValidCandidateOutcome,
    PairedMeasurements,
    RolloutFailure,
    RolloutRecord,
)


def test_summary_includes_failures_in_its_denominator(
    rollout_record: RolloutRecord,
) -> None:
    invalid = rollout_record.model_copy(
        update={"candidates": [], "final": NoValidCandidateOutcome()}
    )
    failed = rollout_record.model_copy(
        update={
            "final": None,
            "failure": RolloutFailure(
                operation="model",
                error_type="ModelError",
                error="unavailable",
                paired=PairedMeasurements(),
            ),
        }
    )
    summary = summarize_performance([rollout_record, invalid, failed])
    assert summary.rollout_count == 3
    assert summary.scored_rollout_count == 1
    assert summary.failure_count == 1
    assert summary.no_valid_candidate_count == 1
    assert summary.geometric_mean_speedup == pytest.approx(2.0)
    assert summary.candidate_workload_time_ms == 5.0
    assert summary.default_workload_time_ms == 10.0
    assert summary.total_workload_speedup == 2.0


def test_empty_summary_has_no_invented_speedup() -> None:
    summary = summarize_performance([])
    assert summary.rollout_count == 0
    assert summary.scored_rollout_count == 0
    assert summary.geometric_mean_speedup is None
    assert summary.total_workload_speedup is None
