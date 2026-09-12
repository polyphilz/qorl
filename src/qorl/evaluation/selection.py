"""Evaluation-only selection from feedback available before final measurement."""

import math
from copy import copy

from qorl.agent.feedback import execution_observation
from qorl.measure.protocols import QueryExecutor
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import ExecutionCounts, PairedMeasurements, SelectionState

MINIMUM_FEEDBACK_SPEEDUP = 1.05


def best_feedback_candidate[ExecutorT: QueryExecutor](
    evaluator: RolloutEvaluator[ExecutorT],
) -> str:
    """Take the earliest best completed eligible probe above the fixed threshold."""
    selected = "default"
    best_ratio = -math.inf
    for candidate in evaluator.candidates:
        if not candidate.selection_eligible or candidate.execution_timed_out:
            continue
        observation = execution_observation(
            candidate, evaluator.default, evaluator.candidates
        )
        ratio = (
            observation.preliminary_ratio_to_initial_default
            if observation is not None
            else None
        )
        if (
            ratio is not None
            and math.isfinite(ratio)
            and ratio >= MINIMUM_FEEDBACK_SPEEDUP
            and ratio > best_ratio
        ):
            selected, best_ratio = candidate.candidate_id, ratio
    return selected


def feedback_selection_evaluator[ExecutorT: QueryExecutor](
    evaluator: RolloutEvaluator[ExecutorT],
) -> tuple[str, RolloutEvaluator[ExecutorT]]:
    """Snapshot the search, sharing its worker but isolating final measurement state.

    Finalization replaces frozen candidates in the list; it never modifies the
    initial baseline, SQL or catalog. Copy before either policy's final timing so
    a subsequent timeout cannot influence the rule or mutate the other result.
    Counts in this branch track additional executions only.
    """
    if evaluator.finalization_started:
        raise RuntimeError("feedback selection must precede final measurement")
    selected = best_feedback_candidate(evaluator)
    alternate = copy(evaluator)
    alternate.candidates = list(evaluator.candidates)
    alternate.selection = SelectionState()
    alternate.paired = PairedMeasurements()
    alternate.execution_counts = ExecutionCounts()
    alternate.kept_default = False
    if selected == "default" and not alternate.candidates:
        alternate.keep_default()
    else:
        alternate.accept_selection(selected)
    return selected, alternate
