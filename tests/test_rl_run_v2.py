from __future__ import annotations

import json
import unittest

from scripts.rl.build_run_v2_inventory import OUTPUT, PILOT, SOURCE, build


class RlRunV2InventoryTest(unittest.TestCase):
    def test_inventory_is_deterministic_balanced_and_new(self) -> None:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        pilot = json.loads(PILOT.read_text(encoding="utf-8"))
        actual = json.loads(OUTPUT.read_text(encoding="utf-8"))

        self.assertEqual(actual, build(source, pilot))
        selected = actual["splits"]["train"]
        prior_ids = {
            item["task_id"]
            for split in pilot["splits"].values()
            for item in split
        }
        self.assertEqual(len(selected), 400)
        self.assertEqual(len({item["task_id"] for item in selected}), 400)
        self.assertFalse({item["task_id"] for item in selected} & prior_ids)
        self.assertEqual(
            sorted(actual["selection"]["template_quotas"].values()),
            [33] * 8 + [34] * 4,
        )
        first_half_counts: dict[str, int] = {}
        for item in selected[:200]:
            template_id = item["template_id"]
            first_half_counts[template_id] = first_half_counts.get(template_id, 0) + 1
        self.assertEqual(set(first_half_counts.values()), {16, 17})
        for index in range(0, len(selected), 4):
            batch = selected[index : index + 4]
            self.assertEqual(len({item["template_id"] for item in batch}), 4)


if __name__ == "__main__":
    unittest.main()
