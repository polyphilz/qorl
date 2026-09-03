from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from qorl.db.exceptions import QueryTimeout
from qorl.db.worker import ExplainResult
from qorl.measure.protocols import QueryExecutor
from qorl.measure.rollout import (
    RIGOROUS_EVALUATION_PROTOCOL_V1,
    RolloutEvaluator,
    measured,
    training_protocol,
)
from qorl.measure.schemas import (
    TIMEOUT_ATTEMPT_PENALTY,
    Baseline,
    Candidate,
    CandidateOutcome,
    MeasurementProtocolId,
    Outcome,
    OutcomeKind,
    score,
)
from qorl.plans.fingerprint import plan_sha256
from qorl.workload.timeouts import TaskTimeout, task_timeout_ms

TASK: dict[str, Any] = {
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


class CalibratedTimeoutWorker(CountingWorker):
    def __init__(self, successful_candidate_executions: int) -> None:
        super().__init__()
        self.successful_candidate_executions = successful_candidate_executions
        self.candidate_executions = 0

    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        if analyze and hint:
            self.candidate_executions += 1
            if self.candidate_executions > self.successful_candidate_executions:
                raise QueryTimeout(timeout_ms)
        result = super().explain(sql, timeout_ms, analyze=analyze, hint=hint)
        if analyze:
            result.document["Execution Time"] = 2_900.0 if hint else 1_000.0
        return result


class TestRollout:
    def test_task_relative_timeout(self) -> None:
        assert task_timeout_ms(10, 120_000) == 5_000
        assert task_timeout_ms(30_000, 120_000) == 90_000
        assert task_timeout_ms(50_000, 120_000) == 120_000

    def test_score_is_clipped(self) -> None:
        assert score(100, 1) == 10.0
        assert score(1, 100) == 0.1

    def test_measurement_records_postgres_buffer_counts(self) -> None:
        result = ExplainResult(
            {
                "Plan": {
                    "Node Type": "Seq Scan",
                    "Shared Hit Blocks": 90,
                    "Shared Read Blocks": 10,
                },
                "Planning Time": 1.0,
                "Execution Time": 2.0,
            },
            "",
        )

        observation = measured(result)

        assert observation.shared_hit_blocks == 90
        assert observation.shared_read_blocks == 10

    def test_invalid_action_consumes_attempt(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,
        )
        evaluator.start()
        result = evaluator.evaluate({"version": 2})
        assert not result.action_valid
        assert result.outcome == CandidateOutcome.MALFORMED
        assert result.errors_or_diagnostics == ["version must equal 1"]
        assert result.attempts_remaining == 4

    def test_candidate_budget_can_be_reduced_for_training(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,
            max_candidates=1,
        )
        evaluator.start()

        result = evaluator.evaluate({"version": 2})

        assert result.attempts_remaining == 0
        with pytest.raises(RuntimeError, match="budget is exhausted"):
            evaluator.evaluate({"version": 2})
        assert (
            evaluator.measurement_protocol.manifest(1)["max_explain_analyze_executions"]
            == 18
        )

    def test_default_plan_duplicate_reuses_measurements(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,
        )
        evaluator.start()
        result = evaluator.evaluate(
            {"version": 1, "scans": [{"relation": "a", "force": "seq"}]}
        )
        assert result.action_valid
        assert result.outcome == CandidateOutcome.DUPLICATE
        assert result.duplicate_of == "default"
        assert result.provisional_speedup == 1.0

    def test_default_fingerprint_winner_is_exactly_one_without_remeasurement(
        self,
    ) -> None:
        worker = Worker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,
        )
        evaluator.start()
        evaluator.evaluate({"version": 1, "scans": [{"relation": "a", "force": "seq"}]})
        analyze_calls_before_finish = worker.analyze_calls

        final = evaluator.finish(random.Random(0))

        assert worker.analyze_calls == analyze_calls_before_finish
        assert final.score == 1.0
        assert final.kind == OutcomeKind.DEFAULT_DUPLICATE
        assert final.score_source == "default_fingerprint"
        assert final.pair_orders == []
        assert (
            final.candidate_median_execution_time_ms
            == final.default_median_execution_time_ms
        )

    def test_keep_default_is_a_zero_reward_terminal_decision(self) -> None:
        worker = Worker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,
        )
        baseline = evaluator.start()
        analyze_calls_before_decision = worker.analyze_calls

        assert evaluator.keep_default() == {"status": "kept_default"}
        final = evaluator.finish(random.Random(0))

        assert worker.analyze_calls == analyze_calls_before_decision
        assert evaluator.candidates == []
        assert final.status == "completed"
        assert final.kind == OutcomeKind.KEPT_DEFAULT
        assert final.decision == "keep_default"
        assert final.winning_candidate_id == "default"
        assert final.winning_plan_sha256 == baseline.plan_sha256
        assert final.score == 1.0
        assert final.trajectory_reward == 0.0
        assert final.pair_orders == []

    def test_keep_default_is_rejected_after_a_candidate(self) -> None:
        evaluator = RolloutEvaluator(
            Worker(),
            Fixture(),
            TASK,
        )
        evaluator.start()
        evaluator.evaluate({"version": 1, "scans": [{"relation": "a", "force": "seq"}]})

        with pytest.raises(RuntimeError, match="before submitting a candidate"):
            evaluator.keep_default()

    def test_rigorous_protocol_remains_26_executions(self) -> None:
        worker = CountingWorker()
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,
        )

        default, candidates, final = self.run_full_rollout(evaluator)

        assert worker.analyze_calls == 26
        assert worker.plain_calls == 6
        assert len(default.measurements) == 3
        assert candidates[0].warmup is not None
        assert len(final.pair_orders) == 5
        assert final.measurement_protocol_id == "rigorous-evaluation-v1"
        assert RIGOROUS_EVALUATION_PROTOCOL_V1.max_explain_analyze_executions == 26

    def test_training_protocol_warms_each_candidate_once(self) -> None:
        worker = CountingWorker()
        protocol = training_protocol(MeasurementProtocolId.RL_TRAINING_V1)
        evaluator = RolloutEvaluator(
            worker,
            Fixture(),
            TASK,
            measurement_protocol=protocol,
        )

        default, candidates, final = self.run_full_rollout(evaluator)

        assert worker.analyze_calls == 18
        assert worker.plain_calls == 6
        assert len(default.measurements) == 1
        assert candidates[0].warmup is not None
        assert len(final.pair_orders) == 3
        assert final.measurement_protocol_id == "rl-training-v1"
        assert protocol.max_explain_analyze_executions == 18
        assert protocol.manifest(1)["max_explain_analyze_executions"] == 10

    def test_serialized_rollout_record_is_unchanged(self) -> None:
        evaluator = RolloutEvaluator(
            CountingWorker(),
            Fixture(),
            TASK,
        )

        default, candidates, final = self.run_full_rollout(evaluator)
        record = {
            "schema_version": 1,
            "task_id": TASK["task_id"],
            "template_id": "test",
            "default": default.to_wire(),
            "candidates": [candidate.to_wire() for candidate in candidates],
            "final": final.to_wire(),
        }

        golden_path = Path(__file__).with_name("golden_rollout.json")
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        assert record == expected

    def test_calibrated_candidate_timeout_is_a_handled_failure(self) -> None:
        default_plan = deepcopy(DEFAULT_PLAN)
        default_plan["Plan Rows"] = 0
        default_plan_sha256 = plan_sha256(default_plan)
        evaluator = RolloutEvaluator(
            TimeoutWorker(),
            Fixture(),
            TASK,
            measurement_protocol=training_protocol(
                MeasurementProtocolId.RL_TRAINING_V2
            ),
            calibrated_timeout=TaskTimeout(
                "job-test", 1_500.0, 5_000, (default_plan_sha256,)
            ),
            timeout_manifest_id="test-timeouts",
        )
        baseline = evaluator.start()

        candidate = evaluator.evaluate(
            {"version": 1, "settings": {"seq_page_cost": 2.0}}
        )
        final = evaluator.finish(random.Random(0))

        assert evaluator.timeout_ms == 5_000
        assert baseline.candidate_timeout is not None
        assert baseline.candidate_timeout.source == "calibrated"
        assert candidate.action_valid
        assert candidate.constraints_satisfied
        assert candidate.execution_timed_out
        assert candidate.outcome == CandidateOutcome.TIMED_OUT
        assert candidate.provisional_speedup == 0.1
        assert final.status == "no_valid_candidate"
        assert final.kind == OutcomeKind.NO_VALID_CANDIDATE
        assert final.invalid_attempt_count == 0
        assert final.timeout_attempt_count == 1

    def test_timeout_scores_use_the_consumed_budget(self) -> None:
        default_plan = deepcopy(DEFAULT_PLAN)
        default_plan["Plan Rows"] = 0
        evaluator = RolloutEvaluator(
            CalibratedTimeoutWorker(successful_candidate_executions=0),
            Fixture(),
            TASK,
            measurement_protocol=training_protocol(
                MeasurementProtocolId.RL_TRAINING_V2
            ),
            calibrated_timeout=TaskTimeout(
                "job-test", 1_000.0, 3_000, (plan_sha256(default_plan),)
            ),
            timeout_manifest_id="test-timeouts",
        )
        evaluator.start()

        candidate = evaluator.evaluate(
            {"version": 1, "settings": {"seq_page_cost": 2.0}}
        )

        assert candidate.execution_timed_out
        assert candidate.provisional_speedup == pytest.approx(1 / 3)

    def test_final_timeout_adds_a_fixed_penalty(self) -> None:
        default_plan = deepcopy(DEFAULT_PLAN)
        default_plan["Plan Rows"] = 0
        evaluator = RolloutEvaluator(
            CalibratedTimeoutWorker(successful_candidate_executions=2),
            Fixture(),
            TASK,
            measurement_protocol=training_protocol(
                MeasurementProtocolId.RL_TRAINING_V2
            ),
            calibrated_timeout=TaskTimeout(
                "job-test", 1_000.0, 3_000, (plan_sha256(default_plan),)
            ),
            timeout_manifest_id="test-timeouts",
        )
        evaluator.start()
        candidate = evaluator.evaluate(
            {"version": 1, "settings": {"seq_page_cost": 2.0}}
        )

        final = evaluator.finish(random.Random(0))

        assert candidate.provisional_speedup == pytest.approx(1_000 / 2_900)
        assert final.status == "candidate_timeout"
        assert final.score == pytest.approx(1 / 3)
        assert final.default_median_execution_time_ms == 1_000
        assert final.invalid_attempt_count == 0
        assert final.timeout_attempt_count == 1
        assert final.trajectory_reward == pytest.approx(
            math.log(1 / 3) - TIMEOUT_ATTEMPT_PENALTY
        )

    def test_calibrated_default_plan_must_still_match(self) -> None:
        protocol = training_protocol(MeasurementProtocolId.RL_TRAINING_V2)
        evaluator = RolloutEvaluator(
            CountingWorker(),
            Fixture(),
            TASK,
            measurement_protocol=protocol,
            calibrated_timeout=TaskTimeout("job-test", 1_500.0, 5_000, ("stale-plan",)),
            timeout_manifest_id="test-timeouts",
        )

        with pytest.raises(RuntimeError, match="default plan differs"):
            evaluator.start()
        assert evaluator.measurement_protocol.protocol_id == "rl-training-v2"
        assert protocol.max_explain_analyze_executions == 18

    @staticmethod
    def run_full_rollout(
        evaluator: RolloutEvaluator[QueryExecutor],
    ) -> tuple[Baseline, list[Candidate], Outcome]:
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
        if not all(candidate.constraints_satisfied for candidate in candidates):
            raise AssertionError("test candidates must all be valid")
        return default, candidates, evaluator.finish(random.Random(0))
