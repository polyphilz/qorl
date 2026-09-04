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


def test_gate_summary_reports_valid_and_novel_plan_rates() -> None:
    summary = summarize(
        [
            rollout("task-1", 1, "candidate", "plan-a"),
            rollout("task-1", 2, "keep_default", None),
            rollout("task-2", 1, "candidate", "plan-b"),
            rollout("task-2", 2, "candidate", "plan-c"),
        ]
    )

    assert summary.task_count == 2
    assert summary.completed_rollouts == 4
    assert summary.failed_rollouts == 0
    assert summary.valid_plan_rate == 0.75
    assert summary.novel_plan_rate == 0.75
