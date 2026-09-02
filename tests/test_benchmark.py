from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qorl.benchmark import load_run_config, summarize


class BenchmarkTest(unittest.TestCase):
    def test_run_config_loads_its_shared_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            policy_path = repository / "configs/policy/test.json"
            policy_path.parent.mkdir(parents=True)
            policy_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "policy": {
                            "type": "random_structured_action",
                            "seed": 7,
                        },
                    }
                )
            )
            run_path = repository / "experiments/000-test/run.json"
            run_path.parent.mkdir(parents=True)
            run_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id_prefix": "test",
                        "policy_config": "configs/policy/test.json",
                    }
                )
            )

            loaded_path, config = load_run_config(repository, str(run_path))

            self.assertEqual(loaded_path, run_path)
            self.assertEqual(config["run_id_prefix"], "test")
            self.assertEqual(config["policy"]["seed"], 7)
            self.assertEqual(config["_policy_config_path"], policy_path)

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
