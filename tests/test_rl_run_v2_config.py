from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


CONFIG = Path("experiments/004-rl-run-v2/train.toml")


class RlRunV2ConfigTest(unittest.TestCase):
    def test_final_run_is_pinned_to_its_inputs_and_limits(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        source = config["orchestrator"]["train"]["source"][0]
        concurrency = config["orchestrator"]["concurrency"]

        self.assertEqual(config["max_steps"], 100)
        self.assertEqual(config["seq_len"], 20_480)
        self.assertEqual(config["run"]["name"], "rl-run-v2")
        self.assertEqual(config["output_dir"], "outputs/rl")
        self.assertFalse(config["clean"])
        self.assertEqual(
            config["model"]["name"], "outputs/rl/rl-pilot-v1-merged"
        )
        self.assertEqual(
            config["env_vars"]["QORL_RL_TIMEOUT_MANIFEST"],
            "experiments/004-rl-run-v2/timeouts.json",
        )
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
        self.assertEqual(
            source["env"]["taskset"]["selection"],
            "experiments/004-rl-run-v2/selection.json",
        )
        self.assertEqual(source["env"]["agent"]["max_total_tokens"], 18_432)
        self.assertEqual(config["inference"]["vllm"]["max_model_len"], 20_480)
        self.assertEqual(config["inference"]["vllm"]["max_num_seqs"], 4)
        self.assertEqual(
            config["ckpt"],
            {"interval": 5, "keep_last": 2, "keep_interval": 10},
        )

        run_dir = Path(config["output_dir"]) / config["run"]["name"]
        self.assertEqual(run_dir, Path("outputs/rl/rl-run-v2"))
        self.assertEqual(
            run_dir / "checkpoints", Path("outputs/rl/rl-run-v2/checkpoints")
        )


if __name__ == "__main__":
    unittest.main()
