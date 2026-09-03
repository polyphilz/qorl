from __future__ import annotations

import json
import random
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from qorl.db.worker import ExplainResult, QueryTimeout
from qorl.measure.rollout import (
    RIGOROUS_EVALUATION_PROTOCOL_V1,
    RL_TRAINING_PROTOCOL_V1,
    RL_TRAINING_PROTOCOL_V2,
    RolloutEvaluator,
    TrainingRolloutEvaluatorV1,
    TrainingRolloutEvaluatorV2,
    score,
    task_timeout_ms,
)
from qorl.plans.fingerprint import plan_sha256
from qorl.workload.timeouts import TaskTimeout

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
        self.analyze_calls = 0

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
            self.analyze_calls += 1
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


class CountingWorker:
    def __init__(self) -> None:
        self.analyze_calls = 0
        self.plain_calls = 0
        self.plan_ids: dict[str, int] = {}

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
        del sql, timeout_ms
        self.analyze_calls += int(analyze)
        self.plain_calls += int(not analyze)
        plan_id = self.plan_ids.setdefault(hint, len(self.plan_ids))
        plan = deepcopy(DEFAULT_PLAN)
        plan["Plan Rows"] = plan_id
        document: dict[str, Any] = {"Plan": plan}
        if analyze:
            document |= {
                "Planning Time": 1.0,
                "Execution Time": 10.0 + plan_id,
            }
        diagnostic = (
            "HintStateDump: {used hints:Set(seq_page_cost)}, "
            "{not used hints:(none)}, {duplicate hints:(none)}, "
            "{error hints:(none)}"
            if hint
            else ""
        )
        return ExplainResult(document, diagnostic)


class TimeoutWorker(CountingWorker):
    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        if analyze and hint:
            raise QueryTimeout(timeout_ms)
        return super().explain(sql, timeout_ms, analyze=analyze, hint=hint)


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
            Worker(),
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )
        evaluator.start()
        result = evaluator.evaluate({"version": 2})
        self.assertFalse(result["action_valid"])
        self.assertEqual(result["attempts_remaining"], 4)

    def test_candidate_budget_can_be_reduced_for_training(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),  # type: ignore[arg-type]
            Fixture(),  # type: ignore[arg-type]
            TASK,
            max_candidates=1,
        )
        evaluator.start()

        result = evaluator.evaluate({"version": 2})

        self.assertEqual(result["attempts_remaining"], 0)
        with self.assertRaisesRegex(RuntimeError, "budget is exhausted"):
            evaluator.evaluate({"version": 2})
        self.assertEqual(
            evaluator.measurement_protocol.manifest(1)[
                "max_explain_analyze_executions"
            ],
            18,
        )

    def test_default_plan_duplicate_reuses_measurements(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )
        evaluator.start()
        result = evaluator.evaluate(
            {"version": 1, "scans": [{"relation": "a", "force": "seq"}]}
        )
        self.assertTrue(result["action_valid"])
        self.assertEqual(result["duplicate_of"], "default")
        self.assertEqual(result["provisional_speedup"], 1.0)

    def test_default_fingerprint_winner_is_exactly_one_without_remeasurement(
        self,
    ) -> None:
        worker = Worker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )
        evaluator.start()
        evaluator.evaluate({"version": 1, "scans": [{"relation": "a", "force": "seq"}]})
        analyze_calls_before_finish = worker.analyze_calls

        final = evaluator.finish(random.Random(0))

        self.assertEqual(worker.analyze_calls, analyze_calls_before_finish)
        self.assertEqual(final["score"], 1.0)
        self.assertEqual(final["score_source"], "default_fingerprint")
        self.assertEqual(final["pair_orders"], [])
        self.assertEqual(
            final["candidate_median_execution_time_ms"],
            final["default_median_execution_time_ms"],
        )

    def test_keep_default_is_a_zero_reward_terminal_decision(self) -> None:
        worker = Worker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )
        baseline = evaluator.start()
        analyze_calls_before_decision = worker.analyze_calls

        self.assertEqual(evaluator.keep_default(), {"status": "kept_default"})
        final = evaluator.finish(random.Random(0))

        self.assertEqual(worker.analyze_calls, analyze_calls_before_decision)
        self.assertEqual(evaluator.candidates, [])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["decision"], "keep_default")
        self.assertEqual(final["winning_candidate_id"], "default")
        self.assertEqual(final["winning_plan_sha256"], baseline["plan_sha256"])
        self.assertEqual(final["score"], 1.0)
        self.assertEqual(final["trajectory_reward"], 0.0)
        self.assertEqual(final["pair_orders"], [])

    def test_keep_default_is_rejected_after_a_candidate(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )
        evaluator.start()
        evaluator.evaluate({"version": 1, "scans": [{"relation": "a", "force": "seq"}]})

        with self.assertRaisesRegex(RuntimeError, "before submitting a candidate"):
            evaluator.keep_default()

    def test_rigorous_protocol_remains_26_executions(self) -> None:
        worker = CountingWorker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )

        default, candidates, final = self.run_full_rollout(evaluator)

        self.assertEqual(worker.analyze_calls, 26)
        self.assertEqual(worker.plain_calls, 6)
        self.assertEqual(len(default["measurements"]), 3)
        self.assertIsNotNone(candidates[0]["warmup"])
        self.assertEqual(len(final["pair_orders"]), 5)
        self.assertEqual(final["measurement_protocol_id"], "rigorous-evaluation-v1")
        self.assertEqual(
            RIGOROUS_EVALUATION_PROTOCOL_V1.max_explain_analyze_executions,
            26,
        )

    def test_training_protocol_uses_13_executions(self) -> None:
        worker = CountingWorker()
        evaluator = TrainingRolloutEvaluatorV1(
            worker,
            Fixture(),
            TASK,  # type: ignore[arg-type]
        )

        default, candidates, final = self.run_full_rollout(evaluator)

        self.assertEqual(worker.analyze_calls, 13)
        self.assertEqual(worker.plain_calls, 6)
        self.assertEqual(len(default["measurements"]), 1)
        self.assertIsNone(candidates[0]["warmup"])
        self.assertEqual(len(final["pair_orders"]), 3)
        self.assertEqual(final["measurement_protocol_id"], "rl-training-v1")
        self.assertEqual(RL_TRAINING_PROTOCOL_V1.max_explain_analyze_executions, 13)

    def test_serialized_rollout_record_is_unchanged(self) -> None:
        evaluator = RolloutEvaluator(
            CountingWorker(),
            Fixture(),  # type: ignore[arg-type]
            TASK,
        )

        default, candidates, final = self.run_full_rollout(evaluator)
        record = {
            "schema_version": 1,
            "task_id": TASK["task_id"],
            "template_id": "test",
            "default": default,
            "candidates": candidates,
            "final": final,
        }

        golden_path = Path(__file__).with_name("golden_rollout.json")
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        assert record == expected

    def test_calibrated_candidate_timeout_is_a_handled_failure(self) -> None:
        default_plan = deepcopy(DEFAULT_PLAN)
        default_plan["Plan Rows"] = 0
        default_plan_sha256 = plan_sha256(default_plan)
        evaluator = TrainingRolloutEvaluatorV2(
            TimeoutWorker(),  # type: ignore[arg-type]
            Fixture(),  # type: ignore[arg-type]
            TASK,
            TaskTimeout("job-test", 1_500.0, 5_000, (default_plan_sha256,)),
            "test-timeouts",
        )
        baseline = evaluator.start()

        candidate = evaluator.evaluate(
            {"version": 1, "settings": {"seq_page_cost": 2.0}}
        )
        final = evaluator.finish(random.Random(0))

        self.assertEqual(evaluator.timeout_ms, 5_000)
        self.assertEqual(baseline["candidate_timeout"]["source"], "calibrated")
        self.assertTrue(candidate["action_valid"])
        self.assertTrue(candidate["constraints_satisfied"])
        self.assertTrue(candidate["execution_timed_out"])
        self.assertEqual(candidate["provisional_speedup"], 0.1)
        self.assertEqual(final["status"], "no_valid_candidate")
        self.assertEqual(final["timeout_attempt_count"], 1)

    def test_calibrated_default_plan_must_still_match(self) -> None:
        evaluator = TrainingRolloutEvaluatorV2(
            CountingWorker(),  # type: ignore[arg-type]
            Fixture(),  # type: ignore[arg-type]
            TASK,
            TaskTimeout("job-test", 1_500.0, 5_000, ("stale-plan",)),
            "test-timeouts",
        )

        with self.assertRaisesRegex(RuntimeError, "default plan differs"):
            evaluator.start()
        self.assertEqual(evaluator.measurement_protocol.protocol_id, "rl-training-v2")
        self.assertEqual(RL_TRAINING_PROTOCOL_V2.max_explain_analyze_executions, 13)

    @staticmethod
    def run_full_rollout(
        evaluator: RolloutEvaluator,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        default = evaluator.start()
        candidates = [
            evaluator.evaluate(
                {
                    "version": 1,
                    "settings": {"seq_page_cost": float(value)},
                }
            )
            for value in range(1, 6)
        ]
        if not all(candidate["constraints_satisfied"] for candidate in candidates):
            raise AssertionError("test candidates must all be valid")
        return default, candidates, evaluator.finish(random.Random(0))
