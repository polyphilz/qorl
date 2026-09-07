from __future__ import annotations

import json
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from qorl.plans.catalog import TaskCatalog
from qorl.plans.exceptions import ActionError
from qorl.plans.schemas import AUTO, JoinMethod, PlanAction, ScanMethod

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
LEADING = {"left": {"left": "a", "right": "b"}, "right": "c"}
JOIN_NAMES = {"hash": "HashJoin", "merge": "MergeJoin", "nestloop": "NestLoop"}
SCAN_NAMES = {
    "seq": "SeqScan",
    "index": "IndexScan",
    "index_only": "IndexOnlyScan",
    "bitmap": "BitmapScan",
}
AUTO_SCAN_HINTS = {
    frozenset({"seq"}): "NoSeqScan",
    frozenset({"index_only"}): "NoIndexOnlyScan",
    frozenset({"bitmap"}): "NoBitmapScan",
    frozenset({"index", "index_only"}): "NoIndexScan",
    frozenset({"seq", "bitmap"}): "IndexOnlyScan",
    frozenset({"seq", "index", "index_only"}): "BitmapScan",
    frozenset({"seq", "index", "bitmap"}): "IndexOnlyScan",
    frozenset({"seq", "index_only", "bitmap"}): "IndexScan",
    frozenset({"index", "index_only", "bitmap"}): "SeqScan",
}


def subsets[MethodT: StrEnum](methods: type[MethodT]) -> list[tuple[MethodT, ...]]:
    return [
        subset
        for size in range(len(methods) + 1)
        for subset in combinations(methods, size)
    ]


def compile_raw(value: Any, catalog: TaskCatalog) -> tuple[dict[str, Any], str]:
    action = PlanAction.from_raw(value, catalog)
    return action.to_wire(), action.compile()


class TestAction:
    def setup_method(self) -> None:
        self.catalog = TaskCatalog.from_task(
            TASK,
            indexes={
                "a": {"table_a_pkey", "table_a_value_idx"},
                "b": {"table_b_pkey"},
                "c": {"table_c_pkey"},
            },
        )

    @pytest.mark.parametrize("force", [AUTO, *JoinMethod])
    @pytest.mark.parametrize("forbid", subsets(JoinMethod))
    def test_normalizes_all_join_method_combinations(
        self, force: str, forbid: tuple[JoinMethod, ...]
    ) -> None:
        raw = {
            "version": 1,
            "joins": [
                {
                    "relations": ["a", "b"],
                    "force": force,
                    "forbid": [method.value for method in forbid],
                }
            ],
        }
        if (
            force in forbid
            or len(forbid) == len(JoinMethod)
            or (force == AUTO and not forbid)
        ):
            with pytest.raises(ActionError):
                PlanAction.from_raw(raw, self.catalog)
            return
        action = PlanAction.from_raw(raw, self.catalog)
        allowed = ({force} if force != AUTO else set(JOIN_NAMES)) - set(forbid)
        assert action.joins[0].allowed_methods == allowed
        expected = (
            JOIN_NAMES[next(iter(allowed))]
            if len(allowed) == 1
            else "No" + JOIN_NAMES[forbid[0]]
        )
        assert action.compile() == f"/*+ {expected}(a b) */"

    @pytest.mark.parametrize("force", [AUTO, *ScanMethod])
    @pytest.mark.parametrize("forbid", subsets(ScanMethod))
    def test_normalizes_all_scan_method_combinations(
        self, force: str, forbid: tuple[ScanMethod, ...]
    ) -> None:
        indexes = (
            ["table_a_value_idx"] if force in {"index", "index_only", "bitmap"} else []
        )
        raw = {
            "version": 1,
            "scans": [
                {
                    "relation": "a",
                    "force": force,
                    "forbid": [method.value for method in forbid],
                    "indexes": indexes,
                }
            ],
        }
        expected = (
            SCAN_NAMES[force]
            if force != AUTO
            else AUTO_SCAN_HINTS.get(frozenset(forbid))
        )
        if force in forbid or len(forbid) == len(ScanMethod) or expected is None:
            with pytest.raises(ActionError):
                PlanAction.from_raw(raw, self.catalog)
            return
        action = PlanAction.from_raw(raw, self.catalog)
        allowed = ({force} if force != AUTO else set(SCAN_NAMES)) - set(forbid)
        assert action.scans[0].allowed_methods == allowed
        assert action.scans[0].indexes == indexes
        arguments = " ".join(["a", *indexes])
        assert action.compile() == f"/*+ {expected}({arguments}) */"

    def test_forbid_index_alone_explains_the_backend_limitation(self) -> None:
        with pytest.raises(ActionError) as caught:
            PlanAction.from_raw(
                {"version": 1, "scans": [{"relation": "a", "forbid": ["index"]}]},
                self.catalog,
            )
        assert str(caught.value) == (
            "scans[0].forbid [index] cannot be expressed by pg_hint_plan while preserving "
            "the other scan choices; NoIndexScan also disables index_only"
        )

    @pytest.mark.parametrize(
        "constraint", [{"force": "hash"}, {"memoize": "forbid"}, {"memoize": "force"}]
    )
    def test_join_and_memoize_targets_must_exist_in_leading(
        self, constraint: dict[str, str]
    ) -> None:
        with pytest.raises(
            ActionError,
            match=r"joins\[0\].relations must be an internal subtree of leading",
        ):
            PlanAction.from_raw(
                {
                    "version": 1,
                    "leading": LEADING,
                    "joins": [{"relations": ["b", "c"], **constraint}],
                },
                self.catalog,
            )

    def test_rows_target_does_not_need_to_exist_in_leading(self) -> None:
        action = PlanAction.from_raw(
            {
                "version": 1,
                "leading": LEADING,
                "row_corrections": [
                    {"relations": ["b", "c"], "mode": "absolute", "value": 1}
                ],
            },
            self.catalog,
        )
        assert action.compile() == "/*+ Leading(((a b) c)) Rows(b c #1) */"

    @pytest.mark.parametrize("force", [AUTO, *JoinMethod])
    @pytest.mark.parametrize("forbid", subsets(JoinMethod))
    def test_memoize_force_requires_an_allowed_nested_loop(
        self, force: str, forbid: tuple[JoinMethod, ...]
    ) -> None:
        raw = {
            "version": 1,
            "joins": [
                {
                    "relations": ["a", "b"],
                    "force": force,
                    "forbid": [method.value for method in forbid],
                    "memoize": "force",
                }
            ],
        }
        if force in forbid or "nestloop" in forbid or force not in {AUTO, "nestloop"}:
            with pytest.raises(ActionError):
                PlanAction.from_raw(raw, self.catalog)
            return
        action = PlanAction.from_raw(raw, self.catalog)
        assert action.compile().endswith("Memoize(a b) */")

    def test_memoize_force_conflicts_with_disabled_memoization(self) -> None:
        with pytest.raises(
            ActionError, match="forces memoization and disables enable_memoize"
        ):
            PlanAction.from_raw(
                {
                    "version": 1,
                    "joins": [{"relations": ["a", "b"], "memoize": "force"}],
                    "settings": {"enable_memoize": False},
                },
                self.catalog,
            )

    def test_inferred_force_conflicts_with_disabled_method(self) -> None:
        with pytest.raises(
            ActionError, match="forces nestloop and disables enable_nestloop"
        ):
            PlanAction.from_raw(
                {
                    "version": 1,
                    "joins": [{"relations": ["a", "b"], "forbid": ["hash", "merge"]}],
                    "settings": {"enable_nestloop": False},
                },
                self.catalog,
            )

    def test_rejects_non_string_forbidden_methods_as_an_action_error(self) -> None:
        with pytest.raises(
            ActionError, match="scans\\[0\\]\\.forbid must contain only"
        ):
            compile_raw(
                {
                    "version": 1,
                    "scans": [{"relation": "a", "forbid": [{"method": "seq"}]}],
                },
                self.catalog,
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
        assert hint == "/*+ Leading(((a b) c)) HashJoin(a b) */"

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
        assert normalized["joins"][0]["relations"] == ["b", "c"]
        assert (
            hint == "/*+ MergeJoin(b c) NoMemoize(b c) "
            "IndexScan(a table_a_value_idx) "
            "DisableIndex(b table_b_pkey) "
            "Rows(b c *10) Parallel(c 2 hard) "
            "Set(enable_hashagg off) Set(random_page_cost 1.1) */"
        )

    def test_rejects_disconnected_join_target(self) -> None:
        with pytest.raises(ActionError, match="not connected"):
            compile_raw(
                {"version": 1, "joins": [{"relations": ["a", "c"]}]},
                self.catalog,
            )

    def test_rejects_leading_that_omits_a_relation(self) -> None:
        with pytest.raises(ActionError, match="every query relation"):
            compile_raw(
                {"version": 1, "leading": {"left": "a", "right": "b"}},
                self.catalog,
            )

    def test_rejects_disconnected_leading_subtree(self) -> None:
        with pytest.raises(ActionError, match="disconnected subtrees"):
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
        with pytest.raises(ActionError, match="both forces and forbids"):
            compile_raw(
                {
                    "version": 1,
                    "scans": [{"relation": "a", "force": "seq", "forbid": ["seq"]}],
                },
                self.catalog,
            )

    def test_rejects_forced_disabled_index(self) -> None:
        with pytest.raises(ActionError, match="both forces and disables"):
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
        with pytest.raises(ActionError, match="unknown fields"):
            compile_raw({"version": 1, "settings": {"work_mem": 1024}}, self.catalog)

    def test_rejects_setting_that_disables_forced_method(self) -> None:
        with pytest.raises(ActionError, match="both forces hash"):
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
        assert "enable_hashjoin" in settings
        assert "enable_group_by_reordering" in settings
        assert "work_mem" not in settings

    def test_tool_schema_matches_golden(self) -> None:
        expected = json.loads(
            (FIXTURES / "plan_action_schema.json").read_text(encoding="utf-8")
        )
        assert PlanAction.tool_schema(["a", "b", "c"]) == expected

    def test_malformed_action_feedback_matches_golden(self) -> None:
        cases = json.loads(
            (FIXTURES / "malformed_actions.json").read_text(encoding="utf-8")
        )
        for case in cases:
            with pytest.raises(ActionError) as raised:
                PlanAction.from_raw(case["action"], self.catalog)
            assert str(raised.value) == case["error"], case["name"]

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
            "from_collapse_limit",
            "join_collapse_limit",
            "enable_async_append",
            "enable_distinct_reordering",
            "enable_parallel_append",
            "enable_partitionwise_aggregate",
            "enable_partitionwise_join",
            "enable_presorted_aggregate",
            "enable_tidscan",
        ):
            assert name not in settings
            with pytest.raises(ActionError, match="unknown fields"):
                compile_raw({"version": 1, "settings": {name: 1}}, self.catalog)

    def test_tid_scan_is_not_exposed_or_accepted(self) -> None:
        scan = PlanAction.tool_schema()["$defs"]["ScanConstraint"]["properties"]
        assert "tid" not in scan["force"]["enum"]
        assert "tid" not in scan["forbid"]["items"]["enum"]
        with pytest.raises(ActionError, match="must be one of"):
            compile_raw(
                {"version": 1, "scans": [{"relation": "a", "force": "tid"}]},
                self.catalog,
            )

    def test_parallel_workers_are_capped_at_two(self) -> None:
        parallel = PlanAction.tool_schema()["$defs"]["ParallelRequest"]
        assert parallel["properties"]["workers"]["maximum"] == 2
        compile_raw(
            {
                "version": 1,
                "parallel": [{"relation": "a", "workers": 2, "mode": "hard"}],
            },
            self.catalog,
        )
        with pytest.raises(ActionError, match="from 0 through 2"):
            compile_raw(
                {
                    "version": 1,
                    "parallel": [{"relation": "a", "workers": 3, "mode": "hard"}],
                },
                self.catalog,
            )
