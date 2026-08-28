from __future__ import annotations

import math
import random
import statistics
from typing import Any

from qorl.action import ActionError, TaskCatalog, compile_action
from qorl.calibration import plan_sha256
from qorl.fixture import JobFixture
from qorl.plan import Verification, compact_plan, hint_status, verify_action
from qorl.worker import ExplainResult, PostgresWorker, WorkerError


DEFAULT_MEASUREMENTS = 3
FINAL_PAIRS = 5
MAX_CANDIDATES = 5
GLOBAL_TIMEOUT_MS = 120_000


def task_timeout_ms(default_median_ms: float, global_cap_ms: int) -> int:
    return min(global_cap_ms, max(5_000, math.ceil(3 * default_median_ms)))


def measured(result: ExplainResult) -> dict[str, Any]:
    plan = result.document["Plan"]
    return {
        "execution_time_ms": result.document["Execution Time"],
        "planning_time_ms": result.document["Planning Time"],
        "plan_sha256": plan_sha256(plan),
    }


def score(default_median_ms: float, candidate_median_ms: float) -> float:
    return min(10.0, max(0.1, default_median_ms / candidate_median_ms))


class RolloutEvaluator:
    def __init__(
        self,
        worker: PostgresWorker,
        fixture: JobFixture,
        task: dict[str, Any],
        *,
        global_timeout_ms: int = GLOBAL_TIMEOUT_MS,
    ) -> None:
        self.worker = worker
        self.task = task
        self.sql = fixture.load_sql(task)
        self.global_timeout_ms = global_timeout_ms
        self.catalog = TaskCatalog.from_task(task, worker.task_indexes(task))
        self.default: dict[str, Any] = {}
        self.candidates: list[dict[str, Any]] = []
        self.by_fingerprint: dict[str, dict[str, Any]] = {}
        self.timeout_ms = global_timeout_ms

    def start(self) -> dict[str, Any]:
        plain = self.worker.explain(self.sql, self.global_timeout_ms)
        warmup = self.worker.explain(
            self.sql, self.global_timeout_ms, analyze=True
        )
        measurements = [
            measured(
                self.worker.explain(
                    self.sql, self.global_timeout_ms, analyze=True
                )
            )
            for _ in range(DEFAULT_MEASUREMENTS)
        ]
        median_ms = statistics.median(
            item["execution_time_ms"] for item in measurements
        )
        self.timeout_ms = task_timeout_ms(median_ms, self.global_timeout_ms)
        self.default = {
            "plan_sha256": plan_sha256(plain.document["Plan"]),
            "plain_explain": plain.document,
            "warmup": measured(warmup),
            "measurements": measurements,
            "median_execution_time_ms": median_ms,
            "compact_plan": compact_plan(warmup.document["Plan"]),
        }
        self.by_fingerprint[self.default["plan_sha256"]] = {
            "candidate_id": "default",
            "measurements": measurements,
            "median_execution_time_ms": median_ms,
        }
        return self.default

    def invalid_candidate(
        self,
        candidate_id: str,
        action: Any,
        error: str,
        *,
        hint: str = "",
        action_valid: bool = False,
        pg_hint_plan: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        result = {
            "candidate_id": candidate_id,
            "action": action,
            "action_valid": action_valid,
            "constraints_satisfied": False,
            "compiled_hint": hint,
            "duplicate_of": None,
            "plan_sha256": None,
            "provisional_measurements": [],
            "provisional_speedup": 0.0,
            "errors_or_diagnostics": [error],
            "pg_hint_plan": pg_hint_plan,
            "attempts_remaining": MAX_CANDIDATES - len(self.candidates) - 1,
        }
        self.candidates.append(result)
        return result

    def evaluate(self, raw_action: Any) -> dict[str, Any]:
        if not self.default:
            raise RuntimeError("rollout baseline has not been started")
        if len(self.candidates) >= MAX_CANDIDATES:
            raise RuntimeError("rollout candidate budget is exhausted")
        candidate_id = f"candidate-{len(self.candidates) + 1:02d}"
        try:
            action, hint = compile_action(raw_action, self.catalog)
        except ActionError as error:
            return self.invalid_candidate(candidate_id, raw_action, str(error))

        try:
            plain = self.worker.explain(
                self.sql, self.timeout_ms, hint=hint
            )
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        verification = verify_action(
            action, plain.document["Plan"], plain.hint_diagnostics
        )
        pg_hint_plan = hint_status(plain.hint_diagnostics)
        if not verification.valid:
            return self.invalid_candidate(
                candidate_id,
                action,
                "; ".join(verification.errors),
                hint=hint,
                action_valid=True,
                pg_hint_plan=pg_hint_plan,
            )

        fingerprint = plan_sha256(plain.document["Plan"])
        duplicate = self.by_fingerprint.get(fingerprint)
        if duplicate is not None:
            result = {
                "candidate_id": candidate_id,
                "action": action,
                "action_valid": True,
                "constraints_satisfied": True,
                "compiled_hint": hint,
                "duplicate_of": duplicate["candidate_id"],
                "plan_sha256": fingerprint,
                "plain_explain": plain.document,
                "compact_plan": compact_plan(plain.document["Plan"]),
                "provisional_measurements": duplicate["measurements"],
                "provisional_median_execution_time_ms": duplicate[
                    "median_execution_time_ms"
                ],
                "provisional_speedup": score(
                    self.default["median_execution_time_ms"],
                    duplicate["median_execution_time_ms"],
                ),
                "errors_or_diagnostics": [],
                "pg_hint_plan": pg_hint_plan,
                "attempts_remaining": MAX_CANDIDATES - len(self.candidates) - 1,
            }
            self.candidates.append(result)
            return result

        try:
            warmup = self.checked_execution(action, hint)
            execution = self.checked_execution(action, hint)
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        observation = measured(execution)
        median_ms = observation["execution_time_ms"]
        result = {
            "candidate_id": candidate_id,
            "action": action,
            "action_valid": True,
            "constraints_satisfied": True,
            "compiled_hint": hint,
            "duplicate_of": None,
            "plan_sha256": fingerprint,
            "plain_explain": plain.document,
            "warmup": measured(warmup),
            "measured_explain_analyze": execution.document,
            "compact_plan": compact_plan(execution.document["Plan"]),
            "provisional_measurements": [observation],
            "provisional_median_execution_time_ms": median_ms,
            "provisional_speedup": score(
                self.default["median_execution_time_ms"], median_ms
            ),
            "errors_or_diagnostics": [],
            "pg_hint_plan": pg_hint_plan,
            "attempts_remaining": MAX_CANDIDATES - len(self.candidates) - 1,
        }
        self.candidates.append(result)
        self.by_fingerprint[fingerprint] = {
            "candidate_id": candidate_id,
            "measurements": [observation],
            "median_execution_time_ms": median_ms,
        }
        return result

    def checked_execution(
        self, action: dict[str, Any], hint: str
    ) -> ExplainResult:
        result = self.worker.explain(
            self.sql, self.timeout_ms, analyze=True, hint=hint
        )
        verification: Verification = verify_action(
            action, result.document["Plan"], result.hint_diagnostics
        )
        if not verification.valid:
            raise WorkerError("; ".join(verification.errors))
        return result

    def finish(self, rng: random.Random) -> dict[str, Any]:
        valid = [
            candidate
            for candidate in self.candidates
            if candidate["action_valid"] and candidate["constraints_satisfied"]
        ]
        invalid_count = len(self.candidates) - len(valid)
        duplicate_count = sum(
            candidate.get("duplicate_of") is not None for candidate in valid
        )
        if not valid:
            return {
                "status": "no_valid_candidate",
                "winning_candidate_id": None,
                "score": 0.0,
                "trajectory_reward": -3.0,
                "invalid_attempt_count": invalid_count,
                "duplicate_attempt_count": duplicate_count,
            }

        winner = min(
            valid,
            key=lambda candidate: candidate[
                "provisional_median_execution_time_ms"
            ],
        )
        action = winner["action"]
        hint = winner["compiled_hint"]
        self.checked_execution(action, hint)
        self.worker.explain(self.sql, self.timeout_ms, analyze=True)

        pair_orders: list[list[str]] = []
        candidate_measurements: list[dict[str, Any]] = []
        default_measurements: list[dict[str, Any]] = []
        for _ in range(FINAL_PAIRS):
            order = ["candidate", "default"]
            rng.shuffle(order)
            pair_orders.append(order)
            for label in order:
                if label == "candidate":
                    candidate_measurements.append(
                        measured(self.checked_execution(action, hint))
                    )
                else:
                    default_measurements.append(
                        measured(
                            self.worker.explain(
                                self.sql, self.timeout_ms, analyze=True
                            )
                        )
                    )

        candidate_median = statistics.median(
            item["execution_time_ms"] for item in candidate_measurements
        )
        default_median = statistics.median(
            item["execution_time_ms"] for item in default_measurements
        )
        final_score = score(default_median, candidate_median)
        return {
            "status": "completed",
            "winning_candidate_id": winner["candidate_id"],
            "winning_plan_sha256": winner["plan_sha256"],
            "pair_orders": pair_orders,
            "candidate_measurements": candidate_measurements,
            "default_measurements": default_measurements,
            "candidate_median_execution_time_ms": candidate_median,
            "default_median_execution_time_ms": default_median,
            "score": final_score,
            "trajectory_reward": (
                math.log(final_score)
                - 0.10 * invalid_count
                - 0.05 * duplicate_count
            ),
            "invalid_attempt_count": invalid_count,
            "duplicate_attempt_count": duplicate_count,
        }
