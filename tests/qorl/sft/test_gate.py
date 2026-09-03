from __future__ import annotations

from qorl.measure.schemas import RunStatus
from qorl.sft.gate import summarize
from qorl.sft.schemas import ActionFamily, GateRollout


def rollout(
    task_id: str, sample: int, decision: str, fingerprint: str | None
) -> GateRollout:
    return GateRollout(
        task_id=task_id,
        template_id="template-1",
        cohort="unlabeled",
        sample=sample,
        seed=sample,
        status=RunStatus.COMPLETED,
        decision=decision,
        action_valid=decision == "candidate",
        constraints_satisfied=decision == "candidate",
        default_duplicate=False,
        novel_fingerprint=fingerprint,
        action_families=[ActionFamily.LEADING] if decision == "candidate" else [],
        error=None,
    )


def test_gate_summary_reports_diversity_and_mixed_decisions() -> None:
    summary = summarize(
        [
            rollout("task-1", 1, "candidate", "plan-a"),
            rollout("task-1", 2, "keep_default", None),
            rollout("task-2", 1, "candidate", "plan-b"),
            rollout("task-2", 2, "candidate", "plan-c"),
        ]
    )

    assert summary.intervention_rate == 0.75
    assert summary.fingerprints_per_intervened_task == 1.5
    assert summary.mixed_decision_task_share == 0.5
    assert summary.leading_constraint_satisfied_rate == 1.0
