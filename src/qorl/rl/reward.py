"""Training-only clipping and ordinary-GRPO rewards; performance evidence stays raw."""

import math

from qorl.measure.schemas import (
    DefaultDuplicateOutcome,
    KeptDefaultOutcome,
    NoValidCandidateOutcome,
    RolloutRecord,
    TimedOutOutcome,
)
from qorl.rl.schemas import ScalarRewardSettings

MIN_TRAINING_SPEEDUP = 0.1
MAX_TRAINING_SPEEDUP = 10.0


def training_speedup(record: RolloutRecord) -> float | None:
    """Clip observed speedup, or derive a timeout proxy from the initial baseline."""
    outcome = record.final
    if outcome is None or isinstance(outcome, NoValidCandidateOutcome):
        return None
    value = (
        outcome.initial_default_median_execution_time_ms / outcome.timeout_ms
        if isinstance(outcome, TimedOutOutcome)
        else outcome.speedup
    )
    return min(MAX_TRAINING_SPEEDUP, max(MIN_TRAINING_SPEEDUP, value))


def scalar_reward(record: RolloutRecord, settings: ScalarRewardSettings) -> float:
    """Compute the configured scalar rule; anchored GRPO bypasses this calculation."""
    outcome = record.final
    if outcome is None:
        raise ValueError("unscored rollout failure cannot receive a reward")
    if isinstance(outcome, NoValidCandidateOutcome):
        return settings.no_valid_candidate_reward
    if isinstance(outcome, KeptDefaultOutcome):
        return 0.0
    speedup = training_speedup(record)
    if speedup is None:
        raise ValueError("selected outcome has no training speedup")
    invalid = sum(
        not item.execution_timed_out and not item.constraints_satisfied
        for item in record.candidates
    )
    duplicates = sum(
        item.constraints_satisfied and item.duplicate_of is not None
        for item in record.candidates
    )
    timeouts = sum(item.execution_timed_out for item in record.candidates)
    quality = 0.0 if isinstance(outcome, DefaultDuplicateOutcome) else math.log(speedup)
    return (
        quality
        - settings.invalid_attempt_penalty * invalid
        - settings.duplicate_attempt_penalty * duplicates
        - settings.timeout_attempt_penalty * timeouts
    )
