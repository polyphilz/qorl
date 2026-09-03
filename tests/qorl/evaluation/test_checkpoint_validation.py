from __future__ import annotations

import unittest
from pathlib import Path

from qorl.evaluation.checkpoint_validation import (
    checkpoint_summary,
    model_command,
    rotated,
)


class RlCheckpointValidationTest(unittest.TestCase):
    def test_policy_order_rotates_without_changing_membership(self) -> None:
        policies = ["start", "step-010", "step-020"]

        self.assertEqual(rotated(policies, 0), policies)
        self.assertEqual(rotated(policies, 1), ["step-010", "step-020", "start"])
        self.assertEqual(rotated(policies, 4), ["step-010", "step-020", "start"])

    def test_model_command_registers_every_adapter(self) -> None:
        command = model_command(
            Path("/venv/vllm"),
            Path("/models/start"),
            {
                10: {"path": "/adapters/10", "rank": 16},
                20: {"path": "/adapters/20", "rank": 16},
            },
            {"tool_call_parser": "qwen3_coder"},
            20_480,
            4,
            8000,
        )

        self.assertIn("--enable-lora", command)
        self.assertIn("step-010=/adapters/10", command)
        self.assertIn("step-020=/adapters/20", command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "4")
        self.assertEqual(command[command.index("--max-loras") + 1], "2")
        self.assertEqual(command[command.index("--max-lora-rank") + 1], "16")

    def test_model_command_rejects_mixed_adapter_ranks(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "different LoRA ranks"):
            model_command(
                Path("/venv/vllm"),
                Path("/models/start"),
                {
                    10: {"path": "/adapters/10", "rank": 8},
                    20: {"path": "/adapters/20", "rank": 16},
                },
                {"tool_call_parser": "qwen3_coder"},
                20_480,
                4,
                8000,
            )

    def test_summary_reports_default_and_novel_behavior(self) -> None:
        result = {
            "status": "completed",
            "task_id": "task-a",
            "default": {"plan_sha256": "same"},
            "final": {
                "status": "completed",
                "trajectory_reward": 0.0,
                "score": 1.0,
                "winning_plan_sha256": "same",
                "candidate_median_execution_time_ms": 10.0,
                "default_median_execution_time_ms": 10.0,
            },
            "candidates": [
                {
                    "action": {"version": 1},
                    "action_valid": True,
                    "constraints_satisfied": True,
                    "duplicate_of": "default",
                    "plan_sha256": "same",
                }
            ],
            "policy_trace": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            "protocol_metrics": {"novel_candidates": 0, "has_novel_candidate": False},
            "database_calls": {"explain": 2, "explain_analyze": 2},
        }

        summary = checkpoint_summary([result], 1)

        self.assertEqual(summary["planned_rollout_count"], 1)
        self.assertEqual(summary["default_winner_rate"], 1.0)
        self.assertEqual(summary["novel_candidate_count"], 0)
        self.assertEqual(summary["rollout_novel_candidate_rate"], 0.0)
