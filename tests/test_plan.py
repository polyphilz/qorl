from __future__ import annotations

import unittest

from qorl.plan import parse_hint_diagnostics, verify_action


DIAGNOSTICS = (
    "NOTICE: pg_hint_plan[qno=0x0]: HintStateDump: "
    "{used hints:SeqScan(a)HashJoin(a b)}, "
    "{not used hints:(none)}, {duplicate hints:(none)}, {error hints:(none)}\n"
)

PLAN = {
    "Node Type": "Hash Join",
    "Plans": [
        {"Node Type": "Seq Scan", "Alias": "a", "Parallel Aware": False},
        {
            "Node Type": "Hash",
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Alias": "b",
                    "Index Name": "b_id_idx",
                    "Parallel Aware": False,
                }
            ],
        },
    ],
}


class PlanTest(unittest.TestCase):
    def test_parses_real_diagnostic_shape(self) -> None:
        diagnostic = parse_hint_diagnostics(DIAGNOSTICS)
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic.used, "SeqScan(a)HashJoin(a b)")
        self.assertEqual(diagnostic.not_used, "(none)")

    def test_verifies_join_tree_join_and_scan(self) -> None:
        result = verify_action(
            {
                "version": 1,
                "leading": {"left": "a", "right": "b"},
                "joins": [
                    {
                        "relations": ["a", "b"],
                        "force": "hash",
                        "forbid": [],
                        "memoize": "auto",
                    }
                ],
                "scans": [
                    {
                        "relation": "b",
                        "force": "index",
                        "forbid": [],
                        "indexes": ["b_id_idx"],
                    }
                ],
            },
            PLAN,
            DIAGNOSTICS,
        )
        self.assertTrue(result.valid, result.errors)

    def test_rejects_unused_hint(self) -> None:
        result = verify_action(
            {"version": 1},
            PLAN,
            DIAGNOSTICS.replace(
                "{not used hints:(none)}", "{not used hints:MergeJoin(a b)}"
            ),
        )
        self.assertFalse(result.valid)
        self.assertIn("not used hints", result.errors[0])

    def test_rejects_actual_plan_mismatch(self) -> None:
        result = verify_action(
            {
                "version": 1,
                "joins": [
                    {
                        "relations": ["a", "b"],
                        "force": "merge",
                        "forbid": [],
                        "memoize": "auto",
                    }
                ],
            },
            PLAN,
            DIAGNOSTICS,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("uses hash, not merge" in error for error in result.errors))

    def test_rejects_disabled_index_in_actual_plan(self) -> None:
        result = verify_action(
            {
                "version": 1,
                "disabled_indexes": [
                    {"relation": "b", "indexes": ["b_id_idx"]}
                ],
            },
            PLAN,
            DIAGNOSTICS,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("uses disabled indexes" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
