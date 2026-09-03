from __future__ import annotations

import json
from pathlib import Path

from qorl.evaluation.live_validation import (
    TRAINING_RUN,
    adapter_path,
    selected_policies,
    summarize,
)


def metrics(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "tool_calls": 3,
        "available_tool_calls": 3,
        "valid_tool_calls": 3,
        "no_tool_call_turns": 0,
        "inspection_calls": 1,
        "valid_inspection_calls": 1,
        "fake_candidate_id_calls": 0,
        "repeated_inspection_calls": 0,
        "finish_calls": 1,
        "keep_default_calls": 0,
        "candidate_attempts": 1,
        "action_valid_candidates": 1,
        "constraint_satisfied_candidates": 1,
        "duplicate_candidates": 1,
        "novel_candidates": 0,
        "has_valid_candidate": True,
        "has_novel_candidate": False,
        "first_candidate_turn": 2,
        "normalized_action_sha256s": ["action"],
        "novel_plan_sha256s": [],
    }
    value.update(overrides)
    return value


class TestLiveProtocolValidation:
    def test_summary_counts_failures_and_completed_rollouts(self) -> None:
        summary = summarize(
            [
                {"status": "completed", "metrics": metrics()},
                {"status": "failed"},
            ]
        )

        assert summary["completed_task_count"] == 1
        assert summary["orchestration_failure_count"] == 1
        assert summary["valid_tool_call_rate"] == 1.0
        assert summary["rollout_valid_candidate_rate"] == 0.5
        assert summary["finish_call_rate"] == 0.5
        assert summary["keep_default_call_rate"] == 0.0

    def test_adapter_path_comes_from_passed_training_report(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / TRAINING_RUN
        adapter = run_dir / "checkpoints/step_7/adapter"
        adapter.mkdir(parents=True)
        for filename in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "qorl-manifest.json",
        ):
            (adapter / filename).touch()
        (run_dir / "training-report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "adapter": "checkpoints/step_7/adapter",
                }
            )
        )

        assert adapter_path(tmp_path) == adapter.resolve()

    def test_policy_selection_defaults_can_be_composed(self) -> None:
        assert [name for name, _ in selected_policies("adapter", "base-first")] == [
            "adapter"
        ]
        assert [name for name, _ in selected_policies("both", "base-first")] == [
            "base",
            "adapter",
        ]
