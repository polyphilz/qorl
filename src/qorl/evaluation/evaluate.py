"""Aggregate measured evaluation evidence independently of RL reward clipping."""

import math

from qorl.evaluation.schemas import PerformanceSummary
from qorl.measure.schemas import (
    DefaultDuplicateOutcome,
    KeptDefaultOutcome,
    MeasuredOutcome,
    OutcomeKind,
    RolloutRecord,
)


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
    )
