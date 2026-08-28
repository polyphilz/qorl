from __future__ import annotations

import unittest

from qorl.calibration import buffers_stable, observation, plan_sha256


def explain(*, rows: int = 10, hits: int = 100, reads: int = 5) -> dict:
    return {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "title",
            "Plan Rows": 100,
            "Actual Rows": rows,
            "Actual Loops": 1,
            "Shared Hit Blocks": hits,
            "Shared Read Blocks": reads,
        },
        "Planning Time": 0.2,
        "Execution Time": 1.5,
    }


class CalibrationTest(unittest.TestCase):
    def test_plan_fingerprint_ignores_runtime_observations(self) -> None:
        first = explain(rows=10, hits=100, reads=5)["Plan"]
        first.update(
            {
                "Cache Hits": 10,
                "Hash Batches": 1,
                "Index Searches": 3,
                "Peak Memory Usage": 12,
                "Rows Removed by Join Filter": 4,
                "Sort Space Used": 8,
                "Workers Launched": 2,
            }
        )
        second = explain(rows=20, hits=200, reads=9)["Plan"]
        second.update(
            {
                "Cache Hits": 100,
                "Hash Batches": 2,
                "Index Searches": 30,
                "Peak Memory Usage": 24,
                "Rows Removed by Join Filter": 40,
                "Sort Space Used": 16,
                "Workers Launched": 1,
            }
        )
        self.assertEqual(plan_sha256(first), plan_sha256(second))

    def test_plan_fingerprint_detects_physical_plan_change(self) -> None:
        first = explain()["Plan"]
        second = {**first, "Node Type": "Index Scan"}
        self.assertNotEqual(plan_sha256(first), plan_sha256(second))

    def test_buffer_stability_requires_same_plan_and_close_counts(self) -> None:
        first = observation(explain(hits=100, reads=5), 1)
        close = observation(explain(hits=101, reads=5), 2)
        far = observation(explain(hits=120, reads=5), 2)
        self.assertTrue(buffers_stable(first, close))
        self.assertFalse(buffers_stable(first, far))


if __name__ == "__main__":
    unittest.main()
