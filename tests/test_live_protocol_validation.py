from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.sft.live_protocol_validation import (
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


class LiveProtocolValidationTest(unittest.TestCase):
    def test_summary_counts_failures_and_completed_rollouts(self) -> None:
        summary = summarize(
            [
                {"status": "completed", "metrics": metrics()},
                {"status": "failed"},
            ]
        )

        self.assertEqual(summary["completed_task_count"], 1)
        self.assertEqual(summary["orchestration_failure_count"], 1)
        self.assertEqual(summary["valid_tool_call_rate"], 1.0)
        self.assertEqual(summary["rollout_valid_candidate_rate"], 0.5)
        self.assertEqual(summary["finish_call_rate"], 0.5)

    def test_adapter_path_comes_from_passed_training_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            run_dir = repository / TRAINING_RUN
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

            self.assertEqual(adapter_path(repository), adapter.resolve())

    def test_policy_selection_defaults_can_be_composed(self) -> None:
        self.assertEqual(
            [name for name, _ in selected_policies("adapter", "base-first")],
            ["adapter"],
        )
        self.assertEqual(
            [name for name, _ in selected_policies("both", "base-first")],
            ["base", "adapter"],
        )


if __name__ == "__main__":
    unittest.main()
