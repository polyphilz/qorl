from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from qorl.rollout import RolloutEvaluator, score, task_timeout_ms
from qorl.worker import ExplainResult


TASK = {
    "task_id": "job-test",
    "sql_path": "queries/test.sql",
    "sql_sha256": "unused",
    "relations": [
        {"alias": "a", "table": "table_a"},
        {"alias": "b", "table": "table_b"},
    ],
    "join_edges": ["a:table_a.id=b:table_b.a_id"],
}

DEFAULT_PLAN = {
    "Node Type": "Hash Join",
    "Plans": [
        {"Node Type": "Seq Scan", "Alias": "a"},
        {
            "Node Type": "Hash",
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Alias": "b",
                    "Index Name": "table_b_pkey",
                }
            ],
        },
    ],
}


class Fixture:
    def load_sql(self, task: dict[str, Any]) -> str:
        return "SELECT 1;"


class Worker:
    def __init__(self) -> None:
        self.times = iter([11.0, 10.0, 12.0, 30.0, 29.0])

    def task_indexes(self, task: dict[str, Any]) -> dict[str, set[str]]:
        return {"a": {"table_a_pkey"}, "b": {"table_b_pkey"}}

    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        document: dict[str, Any] = {"Plan": DEFAULT_PLAN}
        if analyze:
            document |= {
                "Planning Time": 1.0,
                "Execution Time": next(self.times),
            }
        diagnostic = (
            "HintStateDump: {used hints:SeqScan(a)}, "
            "{not used hints:(none)}, {duplicate hints:(none)}, "
            "{error hints:(none)}"
            if hint
            else ""
        )
        return ExplainResult(document, diagnostic)


class RolloutTest(unittest.TestCase):
    def test_task_relative_timeout(self) -> None:
        self.assertEqual(task_timeout_ms(10, 120_000), 5_000)
        self.assertEqual(task_timeout_ms(30_000, 120_000), 90_000)
        self.assertEqual(task_timeout_ms(50_000, 120_000), 120_000)

    def test_score_is_clipped(self) -> None:
        self.assertEqual(score(100, 1), 10.0)
        self.assertEqual(score(1, 100), 0.1)

    def test_invalid_action_consumes_attempt(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(), Fixture(), TASK  # type: ignore[arg-type]
        )
        evaluator.start()
        result = evaluator.evaluate({"version": 2})
        self.assertFalse(result["action_valid"])
        self.assertEqual(result["attempts_remaining"], 4)

    def test_default_plan_duplicate_reuses_measurements(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(), Fixture(), TASK  # type: ignore[arg-type]
        )
        evaluator.start()
        result = evaluator.evaluate({"version": 1, "scans": [{"relation": "a", "force": "seq"}]})
        self.assertTrue(result["action_valid"])
        self.assertEqual(result["duplicate_of"], "default")
        self.assertEqual(result["provisional_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
