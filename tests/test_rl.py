from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from qorl.fixture import sha256_file
from qorl.rl import CONFIG, PRE_RL_REPORT, verify_pre_rl_validation


class RlTest(unittest.TestCase):
    def test_pilot_config_has_twelve_four_group_updates(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(config["max_steps"], 12)
        self.assertEqual(config["orchestrator"]["batch_size"], 16)
        self.assertEqual(config["orchestrator"]["group_size"], 4)
        self.assertEqual(
            config["orchestrator"]["train"]["source"][0]["env"]["taskset"]["split"],
            "train",
        )

    def test_pre_rl_gate_checks_inputs_model_and_reward_variance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            inputs = {
                "config_sha256": repository
                / "experiments/003-rl-pilot-v1/validation.json",
                "selection_sha256": repository / "experiments/003-rl-pilot-v1/selection.json",
                "run_config_sha256": repository / "configs/policy/run-v1.json",
            }
            for path in inputs.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("input")
            report = {
                "status": "completed",
                "phase": "pre",
                **{name: sha256_file(path) for name, path in inputs.items()},
                "model": {"model_safetensors_sha256": "model"},
                "summary": {
                    "completed_rollout_count": 64,
                    "orchestration_failure_count": 0,
                    "task_group_count": 16,
                    "nonzero_reward_variance_group_count": 16,
                },
            }
            path = repository / PRE_RL_REPORT
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report))

            verify_pre_rl_validation(repository, "model")
            report["summary"]["nonzero_reward_variance_group_count"] = 15
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                verify_pre_rl_validation(repository, "model")


if __name__ == "__main__":
    unittest.main()
