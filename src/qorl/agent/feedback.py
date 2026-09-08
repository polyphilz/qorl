"""Present retained probe evidence."""

import statistics

from qorl.agent.presentation import plan_view
from qorl.agent.schemas import (
    CandidateHistory,
    CandidateSummary,
    ExecutionObservation,
)
from qorl.measure.schemas import Baseline, Candidate, SelectionState


def candidate_history(
    candidates: list[Candidate],
    baseline: Baseline | None,
    attempts_remaining: int,
    selection: SelectionState,
) -> CandidateHistory:
    summaries: list[CandidateSummary] = []
    for candidate in candidates:
        feedback = execution_observation(candidate, baseline, candidates)
        summaries.append(
            CandidateSummary(
                candidate_id=candidate.candidate_id,
                action_valid=candidate.action_valid,
                constraints_satisfied=candidate.constraints_satisfied,
                selection_eligible=candidate.selection_eligible,
                timeout_phase=(
                    "planning" if candidate.plain_explain is None else "execution"
                )
                if candidate.execution_timed_out
                else None,
                timeout_ms=candidate.timeout_ms,
                feedback_source_id=feedback.source_id if feedback else None,
                feedback_median_execution_time_ms=feedback.median_execution_time_ms
                if feedback
                else None,
                preliminary_ratio_to_initial_default=feedback.preliminary_ratio_to_initial_default
                if feedback
                else None,
            )
        )
    return CandidateHistory(
        candidates=summaries,
        selectable_candidate_ids=[
            item.candidate_id for item in candidates if item.selection_eligible
        ],
        attempts_remaining=attempts_remaining,
        selection_status=selection.status,
        selected_candidate_id=selection.selected_candidate_id,
    )


def execution_observation(
    candidate: Candidate,
    baseline: Baseline | None,
    candidates: list[Candidate],
    *,
    node_id: str = "0",
    summary: bool = True,
) -> ExecutionObservation | None:
    feedback = candidate.execution_feedback
    if feedback is None:
        return None
    source_id = feedback.source_id or candidate.candidate_id
    warmups, samples = feedback.warmups, feedback.measurements
    if source_id == "default":
        if baseline is None:
            raise RuntimeError("reused default feedback has no baseline")
        warmups, samples = baseline.warmups, baseline.measurements
    elif feedback.source_id is not None:
        source = next(item for item in candidates if item.candidate_id == source_id)
        if source.execution_feedback is None:
            raise RuntimeError("reused candidate feedback has no evidence")
        warmups, samples = (
            source.execution_feedback.warmups,
            source.execution_feedback.measurements,
        )
    median = (
        statistics.median(item.execution_time_ms for item in samples)
        if samples and feedback.status == "completed"
        else None
    )
    displayed = samples or warmups
    document = displayed[-1].analyzed_document if displayed else None
    return ExecutionObservation(
        status=feedback.status,
        source_id=source_id,
        reused=feedback.source_id is not None,
        new_executions=0
        if feedback.source_id is not None
        else len(warmups) + len(samples) + (feedback.status == "timed_out"),
        warmup_count=len(warmups),
        measurement_count=len(samples),
        median_execution_time_ms=median,
        preliminary_ratio_to_initial_default=baseline.median_execution_time_ms / median
        if median is not None
        and median > 0
        and baseline is not None
        and baseline.median_execution_time_ms is not None
        else None,
        timeout_ms=feedback.timeout_ms,
        displayed_sample_phase=("measurement" if samples else "warmup")
        if displayed
        else None,
        displayed_sample_index=len(displayed) - 1 if displayed else None,
        displayed_sample_execution_time_ms=displayed[-1].execution_time_ms
        if displayed
        else None,
        plan=plan_view(
            document["Plan"], node_id=node_id, summary=summary, observed=True
        )
        if document is not None
        else None,
    )
