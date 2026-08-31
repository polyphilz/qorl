from __future__ import annotations

import asyncio
import inspect
import unittest
from types import SimpleNamespace

from qorl_training.taskset import QorlTask, selected_items


class QorlTaskTest(unittest.TestCase):
    def test_known_inventories_only_expose_their_declared_splits(self) -> None:
        run = {
            "inventory_id": "qorl-rl-run-v2",
            "splits": {"train": [{"task_id": "task", "template_id": "template"}]},
        }

        self.assertEqual(selected_items(run, "train"), run["splits"]["train"])
        with self.assertRaisesRegex(ValueError, "not allowed"):
            selected_items(run, "validation")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            selected_items({"inventory_id": "unknown"}, "train")

    def test_scoring_hooks_are_async_and_return_numeric_signals(self) -> None:
        trace = SimpleNamespace(
            calls=[object(), object()],
            num_total_tokens=100,
            num_output_tokens=10,
            info={
                "qorl": {
                    "candidates": [
                        {
                            "constraints_satisfied": True,
                            "duplicate_of": None,
                        },
                        {
                            "constraints_satisfied": False,
                            "duplicate_of": "candidate-01",
                        },
                    ],
                    "final": {
                        "status": "completed",
                        "score": 1.2,
                        "trajectory_reward": 0.15,
                    },
                }
            },
        )

        self.assertTrue(inspect.iscoroutinefunction(QorlTask.trajectory_reward))
        self.assertTrue(inspect.iscoroutinefunction(QorlTask.rollout_metrics))
        reward = asyncio.run(QorlTask.trajectory_reward(None, trace))
        metrics = asyncio.run(QorlTask.rollout_metrics(None, trace))

        self.assertEqual(reward, 0.15)
        self.assertEqual(metrics["candidate_attempts"], 2.0)
        self.assertEqual(metrics["valid_candidate_attempts"], 1.0)
        self.assertEqual(metrics["duplicate_candidate_attempts"], 1.0)


if __name__ == "__main__":
    unittest.main()
