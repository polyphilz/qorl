from __future__ import annotations

import unittest

from qorl.benchmark import summarize


class BenchmarkTest(unittest.TestCase):
    def test_summary_reports_primary_metrics(self) -> None:
        results = [
            {
                "candidates": [
                    {"constraints_satisfied": True, "duplicate_of": None}
                ],
                "final": {
                    "status": "completed",
                    "score": 2.0,
                    "candidate_median_execution_time_ms": 5.0,
                    "default_median_execution_time_ms": 10.0,
                },
            },
            {
                "candidates": [
                    {"constraints_satisfied": False, "duplicate_of": None}
                ],
                "final": {"status": "no_valid_candidate"},
            },
        ]
        summary = summarize(results)
        self.assertEqual(summary["scored_task_count"], 1)
        self.assertEqual(summary["failure_rate"], 0.5)
        self.assertEqual(summary["geometric_mean_speedup"], 2.0)
        self.assertEqual(summary["invalid_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
