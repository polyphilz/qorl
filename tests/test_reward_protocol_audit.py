from __future__ import annotations

import unittest

from qorl.evaluation.reward_protocols import direction, summarize


def result(cheap: float, full: float) -> dict:
    def protocol(score: float, winner: str = "candidate-01") -> dict:
        return {
            "final": {
                "status": "completed",
                "score": score,
                "trajectory_reward": score - 1,
                "winning_candidate_id": winner,
            },
            "database_calls": {"explain_analyze": 1},
        }

    return {
        "rl-training-v1": protocol(cheap),
        "rigorous-evaluation-v1": protocol(full),
    }


class RewardProtocolAuditTest(unittest.TestCase):
    def test_material_direction_uses_a_symmetric_ratio(self) -> None:
        self.assertEqual(direction(1.019, 1.02), "neutral")
        self.assertEqual(direction(1.021, 1.02), "improved")
        self.assertEqual(direction(0.981, 1.02), "neutral")
        self.assertEqual(direction(0.979, 1.02), "regressed")

    def test_summary_distinguishes_strict_and_material_agreement(self) -> None:
        summary = summarize(
            [result(1.01, 0.99), result(1.20, 1.10)], 1.02
        )

        self.assertEqual(summary["strict_speedup_direction_agreement_count"], 1)
        self.assertEqual(summary["material_direction_agreement_count"], 2)
        self.assertEqual(summary["opposite_material_direction_count"], 0)


if __name__ == "__main__":
    unittest.main()
