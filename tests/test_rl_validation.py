from __future__ import annotations

import unittest

from scripts.rl.paired_validation import summarize


def result(task_id: str, reward: float, speedup: float | None) -> dict:
    final = {
        "status": "no_valid_candidate",
        "trajectory_reward": reward,
    }
    if speedup is not None:
        final.update(
            {
                "status": "completed",
                "score": speedup,
                "candidate_median_execution_time_ms": 5.0,
                "default_median_execution_time_ms": 10.0,
            }
        )
    return {
        "status": "completed",
        "task_id": task_id,
        "final": final,
        "candidates": [
            {
                "action": {"version": 1},
                "action_valid": True,
                "constraints_satisfied": speedup is not None,
                "duplicate_of": None,
                "plan_sha256": "plan" if speedup is not None else None,
            }
        ],
        "policy_trace": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        "database_calls": {"explain": 3, "explain_analyze": 2},
    }


class RlValidationTest(unittest.TestCase):
    def test_summary_keeps_invalid_rollouts_in_reward_mean(self) -> None:
        summary = summarize([result("task-a", 0.5, 2.0), result("task-a", -3.0, None)])

        self.assertEqual(summary["completed_rollout_count"], 2)
        self.assertEqual(summary["valid_rollout_count"], 1)
        self.assertEqual(summary["valid_rollout_rate"], 0.5)
        self.assertEqual(summary["mean_trajectory_reward"], -1.25)
        self.assertEqual(summary["geometric_mean_speedup"], 2.0)
        self.assertEqual(summary["total_workload_speedup"], 2.0)
        self.assertEqual(summary["postgres_explain_analyze_call_count"], 4)
        self.assertEqual(summary["nonzero_reward_variance_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
