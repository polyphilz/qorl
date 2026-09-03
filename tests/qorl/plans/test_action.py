from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from qorl.plans.catalog import TaskCatalog
from qorl.plans.exceptions import ActionError
from qorl.plans.schemas import PlanAction

TASK = {
    "relations": [
        {"alias": "a", "table": "table_a"},
        {"alias": "b", "table": "table_b"},
        {"alias": "c", "table": "table_c"},
    ],
    "join_edges": [
        "a:table_a.id=b:table_b.a_id",
        "b:table_b.id=c:table_c.b_id",
    ],
}
FIXTURES = Path(__file__).with_name("fixtures")


def compile_raw(value: Any, catalog: TaskCatalog) -> tuple[dict[str, Any], str]:
    action = PlanAction.from_raw(value, catalog)
    return action.to_wire(), action.compile()


class ActionTest(unittest.TestCase):
    def test_rejects_non_string_forbidden_methods_as_an_action_error(self) -> None:
        with self.assertRaisesRegex(
            ActionError, "scans\\[0\\]\\.forbid must contain only"
        ):
            compile_raw(
                {
                    "version": 1,
                    "scans": [{"relation": "a", "forbid": [{"method": "seq"}]}],
                },
                self.catalog,
            )

    def setUp(self) -> None:
        self.catalog = TaskCatalog.from_task(
            TASK,
            indexes={
                "a": {"table_a_pkey", "table_a_value_idx"},
                "b": {"table_b_pkey"},
                "c": {"table_c_pkey"},
            },
        )

    def test_compiles_plan_example(self) -> None:
        _, hint = compile_raw(
            {
                "version": 1,
                "leading": {
                    "left": {"left": "a", "right": "b"},
                    "right": "c",
                },
                "joins": [
                    {
                        "relations": ["b", "a"],
                        "force": "hash",
                        "memoize": "auto",
                    }
                ],
            },
            self.catalog,
        )
        self.assertEqual(hint, "/*+ Leading(((a b) c)) HashJoin(a b) */")

    def test_compiles_every_hint_family_deterministically(self) -> None:
        normalized, hint = compile_raw(
            {
                "version": 1,
                "scans": [
                    {
                        "relation": "a",
                        "force": "index",
                        "indexes": ["table_a_value_idx"],
                        "forbid": ["seq"],
                    }
                ],
                "disabled_indexes": [{"relation": "b", "indexes": ["table_b_pkey"]}],
                "joins": [
                    {
                        "relations": ["b", "c"],
                        "force": "merge",
                        "forbid": ["hash"],
                        "memoize": "forbid",
                    }
                ],
                "row_corrections": [
                    {"relations": ["c", "b"], "mode": "multiply", "value": 10}
                ],
                "parallel": [{"relation": "c", "workers": 2, "mode": "hard"}],
                "settings": {"enable_hashagg": False, "random_page_cost": 1.1},
            },
            self.catalog,
        )
        self.assertEqual(normalized["joins"][0]["relations"], ["b", "c"])
        self.assertEqual(
            hint,
            "/*+ MergeJoin(b c) NoHashJoin(b c) NoMemoize(b c) "
            "IndexScan(a table_a_value_idx) NoSeqScan(a) "
            "DisableIndex(b table_b_pkey) "
            "Rows(b c *10) Parallel(c 2 hard) "
            "Set(enable_hashagg off) Set(random_page_cost 1.1) */",
        )

    def test_rejects_disconnected_join_target(self) -> None:
        with self.assertRaisesRegex(ActionError, "not connected"):
            compile_raw(
                {"version": 1, "joins": [{"relations": ["a", "c"]}]},
                self.catalog,
            )

    def test_rejects_leading_that_omits_a_relation(self) -> None:
        with self.assertRaisesRegex(ActionError, "every query relation"):
            compile_raw(
                {"version": 1, "leading": {"left": "a", "right": "b"}},
                self.catalog,
            )

    def test_rejects_disconnected_leading_subtree(self) -> None:
        with self.assertRaisesRegex(ActionError, "disconnected subtrees"):
            compile_raw(
                {
                    "version": 1,
                    "leading": {
                        "left": {"left": "a", "right": "c"},
                        "right": "b",
                    },
                },
                self.catalog,
            )

    def test_rejects_scan_conflict(self) -> None:
        with self.assertRaisesRegex(ActionError, "both forces and forbids"):
            compile_raw(
                {
                    "version": 1,
                    "scans": [{"relation": "a", "force": "seq", "forbid": ["seq"]}],
                },
                self.catalog,
            )

    def test_rejects_forced_disabled_index(self) -> None:
        with self.assertRaisesRegex(ActionError, "both forces and disables"):
            compile_raw(
                {
                    "version": 1,
                    "scans": [
                        {
                            "relation": "a",
                            "force": "index",
                            "indexes": ["table_a_pkey"],
                        }
                    ],
                    "disabled_indexes": [
                        {"relation": "a", "indexes": ["table_a_pkey"]}
                    ],
                },
                self.catalog,
            )

    def test_rejects_unallowlisted_setting(self) -> None:
        with self.assertRaisesRegex(ActionError, "unknown fields"):
            compile_raw({"version": 1, "settings": {"work_mem": 1024}}, self.catalog)

    def test_rejects_setting_that_disables_forced_method(self) -> None:
        with self.assertRaisesRegex(ActionError, "both forces hash"):
            compile_raw(
                {
                    "version": 1,
                    "joins": [{"relations": ["a", "b"], "force": "hash"}],
                    "settings": {"enable_hashjoin": False},
                },
                self.catalog,
            )

    def test_model_schema_matches_the_setting_allowlist(self) -> None:
        settings = PlanAction.tool_schema()["$defs"]["PlannerSettings"]["properties"]
        self.assertIn("enable_hashjoin", settings)
        self.assertNotIn("work_mem", settings)

    def test_tool_schema_matches_golden(self) -> None:
        expected = json.loads(
            (FIXTURES / "plan_action_schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            PlanAction.tool_schema(["a", "b", "c"]),
            expected,
        )

    def test_malformed_action_feedback_matches_golden(self) -> None:
        cases = json.loads(
            (FIXTURES / "malformed_actions.json").read_text(encoding="utf-8")
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(ActionError) as raised:
                    PlanAction.from_raw(case["action"], self.catalog)
                self.assertEqual(str(raised.exception), case["error"])

    def test_removed_settings_are_not_exposed_or_accepted(self) -> None:
        settings = PlanAction.tool_schema()["$defs"]["PlannerSettings"]["properties"]
        for name in (
            "effective_io_concurrency",
            "max_parallel_workers_per_gather",
            "enable_partition_pruning",
            "geqo",
            "geqo_threshold",
            "geqo_effort",
            "geqo_pool_size",
            "geqo_generations",
            "geqo_selection_bias",
            "geqo_seed",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, settings)
                with self.assertRaisesRegex(ActionError, "unknown fields"):
                    compile_raw({"version": 1, "settings": {name: 1}}, self.catalog)

    def test_parallel_workers_are_capped_at_two(self) -> None:
        parallel = PlanAction.tool_schema()["$defs"]["ParallelRequest"]
        self.assertEqual(parallel["properties"]["workers"]["maximum"], 2)
        compile_raw(
            {
                "version": 1,
                "parallel": [{"relation": "a", "workers": 2, "mode": "hard"}],
            },
            self.catalog,
        )
        with self.assertRaisesRegex(ActionError, "from 0 through 2"):
            compile_raw(
                {
                    "version": 1,
                    "parallel": [{"relation": "a", "workers": 3, "mode": "hard"}],
                },
                self.catalog,
            )
