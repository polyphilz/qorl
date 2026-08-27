from __future__ import annotations

import random
import unittest

from qprl.action import TaskCatalog, normalize_action
from qprl.random_policy import FAMILY_WEIGHTS, sample_action


TASK = {
    "relations": [
        {"alias": "a", "table": "table_a"},
        {"alias": "b", "table": "table_b"},
        {"alias": "c", "table": "table_c"},
        {"alias": "d", "table": "table_d"},
    ],
    "join_edges": [
        "a:table_a.id=b:table_b.a_id",
        "b:table_b.id=c:table_c.b_id",
        "b:table_b.id=d:table_d.b_id",
    ],
}


class RandomPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = TaskCatalog.from_task(
            TASK,
            indexes={relation: {f"table_{relation}_pkey"} for relation in "abcd"},
        )

    def test_sampler_is_deterministic_and_schema_valid(self) -> None:
        left = random.Random(1234)
        right = random.Random(1234)
        for _ in range(1_000):
            action = sample_action(self.catalog, left)
            self.assertEqual(action, sample_action(self.catalog, right))
            self.assertEqual(action, normalize_action(action, self.catalog))
            self.assertLessEqual(len(action) - 1, 3)

    def test_fixed_sample_reaches_every_family(self) -> None:
        rng = random.Random(20260827)
        observed: set[str] = set()
        field_to_family = {
            "leading": "leading",
            "joins": "join",
            "scans": "scan",
            "row_corrections": "rows",
            "parallel": "parallel",
            "settings": "setting",
        }
        for _ in range(2_000):
            action = sample_action(self.catalog, rng)
            observed.update(field_to_family[field] for field in action if field != "version")
        self.assertEqual(observed, set(FAMILY_WEIGHTS))


if __name__ == "__main__":
    unittest.main()
