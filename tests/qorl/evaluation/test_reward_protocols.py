from __future__ import annotations

from qorl.evaluation.reward_protocols import direction, summarize


def result(cheap: float, full: float) -> dict:
    def protocol(
        score: float,
        *,
        repeat_count: int,
        retained_material_count: int,
        winner: str = "candidate-01",
    ) -> dict:
        return {
            "final": {
                "status": "completed",
                "score": score,
                "trajectory_reward": score - 1,
                "winning_candidate_id": winner,
            },
            "database_calls": {"explain_analyze": 1},
            "cold_read_guard": {
                "repeat_count": repeat_count,
                "initial_candidate_measurement_count": 1,
                "retained_candidate_measurement_count": 1,
                "retained_material_shared_read_count": retained_material_count,
            },
        }

    return {
        "rl-training-v1": protocol(
            cheap,
            repeat_count=1,
            retained_material_count=0,
        ),
        "rigorous-evaluation-v1": protocol(
            full,
            repeat_count=0,
            retained_material_count=1,
        ),
    }


class TestRewardProtocolAudit:
    def test_material_direction_uses_a_symmetric_ratio(self) -> None:
        assert direction(1.019, 1.02) == "neutral"
        assert direction(1.021, 1.02) == "improved"
        assert direction(0.981, 1.02) == "neutral"
        assert direction(0.979, 1.02) == "regressed"

    def test_summary_distinguishes_strict_and_material_agreement(self) -> None:
        summary = summarize([result(1.01, 0.99), result(1.20, 1.10)], 1.02)

        assert summary["strict_speedup_direction_agreement_count"] == 1
        assert summary["material_direction_agreement_count"] == 2
        assert summary["opposite_material_direction_count"] == 0
        assert summary["cold_read_audit"] == {
            "material_shared_read_fraction": 0.10,
            "rl-training-v1": {
                "guard_repeat_count": 2,
                "guard_repeat_rate": 1.0,
                "initial_candidate_measurement_count": 2,
                "retained_candidate_measurement_count": 2,
                "retained_material_shared_read_count": 0,
                "retained_material_shared_read_rate": 0.0,
            },
            "rigorous-evaluation-v1": {
                "guard_repeat_count": 0,
                "guard_repeat_rate": 0.0,
                "initial_candidate_measurement_count": 2,
                "retained_candidate_measurement_count": 2,
                "retained_material_shared_read_count": 2,
                "retained_material_shared_read_rate": 1.0,
            },
        }
