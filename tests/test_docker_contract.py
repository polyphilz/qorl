from __future__ import annotations

import json
import unittest
from pathlib import Path

from qorl.action import BOOLEAN_SETTINGS, INTEGER_SETTINGS, NUMERIC_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


class DockerContractTest(unittest.TestCase):
    def test_contract_defines_every_prompt_visible_planner_setting(self) -> None:
        contract = json.loads(
            (
                ROOT
                / "docker/postgres/contract/benchmark.expected.json"
            ).read_text(encoding="utf-8")
        )
        names = set(BOOLEAN_SETTINGS) | set(INTEGER_SETTINGS) | set(NUMERIC_SETTINGS)
        values = {name: contract["settings"][name] for name in sorted(names)}

        self.assertEqual(set(values), names)
        self.assertTrue(all(isinstance(value, str) for value in values.values()))

    def test_benchmark_v2_disables_geqo(self) -> None:
        contract = json.loads(
            (
                ROOT
                / "docker/postgres/contract/benchmark.expected.json"
            ).read_text(encoding="utf-8")
        )
        versions = json.loads(
            (ROOT / "docker/postgres/versions.json").read_text(encoding="utf-8")
        )

        self.assertEqual(contract["benchmark_config_id"], "benchmark-v2")
        self.assertEqual(versions["benchmark"]["config_id"], "benchmark-v2")
        self.assertEqual(contract["settings"]["geqo"], "off")


if __name__ == "__main__":
    unittest.main()
