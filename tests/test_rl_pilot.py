from __future__ import annotations

import json
import unittest

from scripts.rl.build_pilot_inventory import OUTPUT, SOURCE, build


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
