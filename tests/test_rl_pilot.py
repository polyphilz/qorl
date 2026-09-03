from __future__ import annotations

import json
import runpy
import unittest


BUILDER = runpy.run_path("experiments/003-rl-pilot-v1/build_inventory.py")
OUTPUT = BUILDER["OUTPUT"]
SOURCE = BUILDER["SOURCE"]
build = BUILDER["build"]


class RlPilotInventoryTest(unittest.TestCase):
    def test_checked_in_inventory_is_deterministic_and_template_balanced(self) -> None:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        expected = build(source)
        actual = json.loads(OUTPUT.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)
        self.assertEqual(actual["counts"], {"spike": 1, "train": 48, "validation": 16})
        self.assertEqual(
            {item["template_id"] for item in actual["splits"]["train"]},
            {task["template_id"] for task in source["tasks"] if task["partition"] == "train"},
        )
        self.assertEqual(
            {item["template_id"] for item in actual["splits"]["validation"]},
            {
                task["template_id"]
                for task in source["tasks"]
                if task["partition"] == "validation"
            },
        )


if __name__ == "__main__":
    unittest.main()
