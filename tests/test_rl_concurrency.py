from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


CONFIG = Path("experiments/004-rl-run-v2/concurrency-spike.toml")


class RlConcurrencyConfigTest(unittest.TestCase):
    def test_spike_matches_four_rollouts_to_four_database_workers(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        source = config["orchestrator"]["train"]["source"][0]
        concurrency = config["orchestrator"]["concurrency"]

        self.assertEqual(config["max_steps"], 3)
        self.assertEqual(config["orchestrator"]["group_size"], 4)
        self.assertEqual(config["orchestrator"]["batch_size"], 16)
        self.assertEqual(config["orchestrator"]["max_off_policy_steps"], 1)
        self.assertEqual(
            (
                concurrency["initial_inflight"],
                concurrency["min_inflight"],
                concurrency["max_inflight"],
            ),
            (4, 4, 4),
        )
        self.assertEqual(source["serve"]["max_concurrent"], 4)
        self.assertEqual(source["serve"]["pool"]["num_workers"], 1)
        self.assertEqual(config["inference"]["vllm"]["max_num_seqs"], 4)
        self.assertEqual(
            source["env"]["taskset"]["selection"],
            "experiments/004-rl-run-v2/selection.json",
        )


if __name__ == "__main__":
    unittest.main()
