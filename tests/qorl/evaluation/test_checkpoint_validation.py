from __future__ import annotations

import unittest
from pathlib import Path

from qorl.evaluation.checkpoint_validation import (
    checkpoint_summary,
    load_config,
    model_command,
    policy_name,
    rotated,
)


class RlCheckpointValidationTest(unittest.TestCase):
    def test_configuration_pins_the_frozen_cohort_and_checkpoint_cadence(self) -> None:
        config, _ = load_config(Path.cwd())

        self.assertEqual(config["selection"], "experiments/003-rl-pilot-v1/selection.json")
        self.assertEqual(config["split"], "validation")
        self.assertEqual(
            config["rollout_seeds"],
            [2026083100, 2026083101, 2026083102, 2026083103],
        )
        self.assertEqual(config["checkpoint_steps"], list(range(10, 101, 10)))
        self.assertEqual(config["concurrency"], 4)

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
                10: {"path": "/adapters/10"},
                20: {"path": "/adapters/20"},
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
