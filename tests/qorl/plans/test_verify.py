from __future__ import annotations

import pytest

from qorl.plans.catalog import TaskCatalog
from qorl.plans.schemas import PlanAction
from qorl.plans.verify import parse_hint_diagnostics, verify_action

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


class TestPlan:
    @pytest.mark.parametrize("node_type", ["Index Scan", "Index Only Scan"])
    @pytest.mark.parametrize("forced", [False, True])
    def test_index_only_fallback_is_valid_only_when_the_full_constraint_allows_it(
        self, node_type: str, forced: bool
    ) -> None:
        catalog = TaskCatalog.from_task(
            {"relations": [{"alias": "a", "table": "table_a"}], "join_edges": []},
            indexes={"a": {"a_idx"}},
        )
        request = (
            {
                "relation": "a",
                "force": "index_only",
                "forbid": ["index"],
                "indexes": ["a_idx"],
            }
            if forced
            else {"relation": "a", "forbid": ["seq", "bitmap"]}
        )
        action = PlanAction.from_raw({"version": 1, "scans": [request]}, catalog)
        plan = {"Node Type": node_type, "Alias": "a", "Index Name": "a_idx"}
        result = verify_action(action.to_wire(), plan, DIAGNOSTICS)
        assert result.valid == (not forced or node_type == "Index Only Scan")
        if not result.valid:
            assert result.errors == (
                "scan a uses index, not index_only",
                "scan a uses forbidden method index",
            )

    @pytest.mark.parametrize(
        "node_type", ["Seq Scan", "Bitmap Heap Scan", "Index Scan", "Index Only Scan"]
    )
    def test_no_index_constraint_checks_both_physical_index_methods(
        self, node_type: str
    ) -> None:
        result = verify_action(
            {
                "version": 1,
                "scans": [
                    {
                        "relation": "a",
                        "force": "auto",
                        "forbid": ["index", "index_only"],
                        "indexes": [],
                    }
                ],
            },
            {"Node Type": node_type, "Alias": "a"},
            DIAGNOSTICS,
        )
        assert result.valid == (node_type in {"Seq Scan", "Bitmap Heap Scan"})

    def test_parses_real_diagnostic_shape(self) -> None:
        diagnostic = parse_hint_diagnostics(DIAGNOSTICS)
        assert diagnostic is not None
        assert diagnostic is not None
        assert diagnostic.used == "SeqScan(a)HashJoin(a b)"
        assert diagnostic.not_used == "(none)"

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
        assert result.valid, result.errors

    def test_rejects_unused_hint(self) -> None:
        result = verify_action(
            {"version": 1},
            PLAN,
            DIAGNOSTICS.replace(
                "{not used hints:(none)}", "{not used hints:MergeJoin(a b)}"
            ),
        )
        assert not result.valid
        assert "not used hints" in result.errors[0]

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
        assert not result.valid
        assert any("uses hash, not merge" in error for error in result.errors)

    def test_memoize_constraint_checks_only_the_join_inner_child(self) -> None:
        plan = {
            "Node Type": "Nested Loop",
            "Plans": [
                {
                    "Node Type": "Nested Loop",
                    "Plans": [
                        {"Node Type": "Seq Scan", "Alias": "a"},
                        {
                            "Node Type": "Memoize",
                            "Plans": [{"Node Type": "Index Scan", "Alias": "b"}],
                        },
                    ],
                },
                {"Node Type": "Index Scan", "Alias": "c"},
            ],
        }
        action = {
            "version": 1,
            "joins": [
                {
                    "relations": ["a", "b", "c"],
                    "force": "auto",
                    "forbid": [],
                    "memoize": "forbid",
                }
            ],
        }

        result = verify_action(action, plan, DIAGNOSTICS)

        assert result.valid, result.errors

    def test_memoize_constraint_detects_a_memoized_inner_child(self) -> None:
        plan = {
            "Node Type": "Nested Loop",
            "Plans": [
                {"Node Type": "Seq Scan", "Alias": "a"},
                {
                    "Node Type": "Memoize",
                    "Plans": [{"Node Type": "Index Scan", "Alias": "b"}],
                },
            ],
        }
        action = {
            "version": 1,
            "joins": [
                {
                    "relations": ["a", "b"],
                    "force": "auto",
                    "forbid": [],
                    "memoize": "force",
                }
            ],
        }

        result = verify_action(action, plan, DIAGNOSTICS)

        assert result.valid, result.errors

    def test_used_memoize_hint_still_requires_physical_inner_child(self) -> None:
        diagnostics = DIAGNOSTICS.replace("SeqScan(a)HashJoin(a b)", "Memoize(a b)")
        assert "{used hints:Memoize(a b)}" in diagnostics
        result = verify_action(
            {
                "version": 1,
                "joins": [
                    {
                        "relations": ["a", "b"],
                        "force": "auto",
                        "forbid": [],
                        "memoize": "force",
                    }
                ],
            },
            {
                "Node Type": "Nested Loop",
                "Plans": [
                    {"Node Type": "Seq Scan", "Alias": "a"},
                    {"Node Type": "Seq Scan", "Alias": "b"},
                ],
            },
            diagnostics,
        )
        assert not result.valid
        assert any("even when the hint is used" in error for error in result.errors)
        assert any(
            "remove the memoization requirement" in error for error in result.errors
        )

    def test_rejects_disabled_index_in_actual_plan(self) -> None:
        result = verify_action(
            {
                "version": 1,
                "disabled_indexes": [{"relation": "b", "indexes": ["b_id_idx"]}],
            },
            PLAN,
            DIAGNOSTICS,
        )
        assert not result.valid
        assert any("uses disabled indexes" in error for error in result.errors)
