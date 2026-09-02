from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

from qorl.action import ActionError, TaskCatalog, compile_action
from qorl.calibration import plan_sha256
from qorl.fixture import TaskSet
from qorl.plan import Verification, compact_plan, hint_status, verify_action
from qorl.timeouts import GLOBAL_TIMEOUT_MS, TaskTimeout, task_timeout_ms
from qorl.worker import (
    ExplainResult,
    PostgresWorker,
    QueryTimeout,
    WorkerError,
)

DEFAULT_MEASUREMENTS = 3
FINAL_PAIRS = 5
MAX_CANDIDATES = 5


@dataclass(frozen=True)
class MeasurementProtocol:
    protocol_id: str
    default_warmup_runs: int
    default_measurement_runs: int
    candidate_warmup_runs: int
    candidate_measurement_runs: int
    final_warmup_pairs: int
    final_randomized_pairs: int

    @property
    def max_explain_analyze_executions(self) -> int:
        return self.max_explain_analyze_executions_for(MAX_CANDIDATES)

    def max_explain_analyze_executions_for(self, candidate_attempts: int) -> int:
        return (
            self.default_warmup_runs
            + self.default_measurement_runs
            + candidate_attempts
            * (self.candidate_warmup_runs + self.candidate_measurement_runs)
            + 2 * self.final_warmup_pairs
            + 2 * self.final_randomized_pairs
        )

    def manifest(
        self, candidate_attempts: int = MAX_CANDIDATES
    ) -> dict[str, int | str]:
        return {
            "id": self.protocol_id,
            "candidate_attempts": candidate_attempts,
            "default_warmup_runs": self.default_warmup_runs,
            "default_measurement_runs": self.default_measurement_runs,
            "novel_candidate_warmup_runs": self.candidate_warmup_runs,
            "novel_candidate_measurement_runs": self.candidate_measurement_runs,
            "final_warmup_runs_per_plan": self.final_warmup_pairs,
            "final_randomized_pair_count": self.final_randomized_pairs,
            "max_explain_analyze_executions": (
                self.max_explain_analyze_executions_for(candidate_attempts)
            ),
        }


RIGOROUS_EVALUATION_PROTOCOL_V1 = MeasurementProtocol(
    protocol_id="rigorous-evaluation-v1",
    default_warmup_runs=1,
    default_measurement_runs=DEFAULT_MEASUREMENTS,
    candidate_warmup_runs=1,
    candidate_measurement_runs=1,
    final_warmup_pairs=1,
    final_randomized_pairs=FINAL_PAIRS,
)

RL_TRAINING_PROTOCOL_V1 = MeasurementProtocol(
    protocol_id="rl-training-v1",
    default_warmup_runs=1,
    default_measurement_runs=1,
    candidate_warmup_runs=0,
    candidate_measurement_runs=1,
    final_warmup_pairs=0,
    final_randomized_pairs=3,
)

RL_TRAINING_PROTOCOL_V2 = MeasurementProtocol(
    protocol_id="rl-training-v2",
    default_warmup_runs=1,
    default_measurement_runs=1,
    candidate_warmup_runs=0,
    candidate_measurement_runs=1,
    final_warmup_pairs=0,
    final_randomized_pairs=3,
)


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
        task_set: TaskSet,
        task: dict[str, Any],
        *,
        global_timeout_ms: int = GLOBAL_TIMEOUT_MS,
        measurement_protocol: MeasurementProtocol = (RIGOROUS_EVALUATION_PROTOCOL_V1),
        calibrated_timeout: TaskTimeout | None = None,
        timeout_manifest_id: str | None = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self.worker = worker
        self.task = task
        self.sql = task_set.load_sql(task)
        self.global_timeout_ms = global_timeout_ms
        self.measurement_protocol = measurement_protocol
        self.catalog = TaskCatalog.from_task(task, worker.task_indexes(task))
        self.default: dict[str, Any] = {}
        self.candidates: list[dict[str, Any]] = []
        self.by_fingerprint: dict[str, dict[str, Any]] = {}
        self.kept_default = False
        self.calibrated_timeout = calibrated_timeout
        self.timeout_manifest_id = timeout_manifest_id
        self.max_candidates = max_candidates
        self.timeout_ms = (
            calibrated_timeout.timeout_ms
            if calibrated_timeout is not None
            else global_timeout_ms
        )

    def start(self) -> dict[str, Any]:
        baseline_timeout_ms = (
            self.timeout_ms
            if self.calibrated_timeout is not None
            else self.global_timeout_ms
        )
        plain = self.worker.explain(self.sql, baseline_timeout_ms)
        default_plan_sha256 = plan_sha256(plain.document["Plan"])
        if (
            self.calibrated_timeout is not None
            and default_plan_sha256 not in self.calibrated_timeout.plan_sha256s
        ):
            raise RuntimeError(
                "default plan differs from calibrated plan: "
                f"task={self.task['task_id']} "
                f"actual={default_plan_sha256}"
            )
        warmups = [
            self.worker.explain(self.sql, baseline_timeout_ms, analyze=True)
            for _ in range(self.measurement_protocol.default_warmup_runs)
        ]
        measurements = [
            measured(self.worker.explain(self.sql, baseline_timeout_ms, analyze=True))
            for _ in range(self.measurement_protocol.default_measurement_runs)
        ]
        median_ms = statistics.median(
            item["execution_time_ms"] for item in measurements
        )
        if self.calibrated_timeout is None:
            self.timeout_ms = task_timeout_ms(median_ms, self.global_timeout_ms)
        self.default = {
            "measurement_protocol_id": self.measurement_protocol.protocol_id,
            "plan_sha256": default_plan_sha256,
            "plain_explain": plain.document,
            "warmup": measured(warmups[-1]) if warmups else None,
            "measurements": measurements,
            "median_execution_time_ms": median_ms,
            "candidate_timeout": {
                "timeout_ms": self.timeout_ms,
                "source": (
                    "calibrated"
                    if self.calibrated_timeout is not None
                    else "live_default"
                ),
                "manifest_id": self.timeout_manifest_id,
                "calibrated_default_median_ms": (
                    self.calibrated_timeout.calibrated_default_ms
                    if self.calibrated_timeout is not None
                    else None
                ),
            },
            "compact_plan": compact_plan(
                (warmups[-1] if warmups else plain).document["Plan"]
            ),
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
        execution_timed_out: bool = False,
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
            "provisional_speedup": 0.1 if execution_timed_out else 0.0,
            "execution_timed_out": execution_timed_out,
            "timeout_ms": self.timeout_ms if execution_timed_out else None,
            "errors_or_diagnostics": [error],
            "pg_hint_plan": pg_hint_plan,
            "attempts_remaining": self.max_candidates - len(self.candidates) - 1,
        }
        self.candidates.append(result)
        return result

    def evaluate(self, raw_action: Any) -> dict[str, Any]:
        if not self.default:
            raise RuntimeError("rollout baseline has not been started")
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        if len(self.candidates) >= self.max_candidates:
            raise RuntimeError("rollout candidate budget is exhausted")
        candidate_id = f"candidate-{len(self.candidates) + 1:02d}"
        try:
            action, hint = compile_action(raw_action, self.catalog)
        except ActionError as error:
            return self.invalid_candidate(candidate_id, raw_action, str(error))

        try:
            plain = self.worker.explain(self.sql, self.timeout_ms, hint=hint)
        except QueryTimeout as error:
            return self.invalid_candidate(
                candidate_id,
                action,
                str(error),
                hint=hint,
                action_valid=True,
                execution_timed_out=True,
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
                "attempts_remaining": self.max_candidates - len(self.candidates) - 1,
            }
            self.candidates.append(result)
            return result

        try:
            warmups = [
                self.checked_execution(action, hint)
                for _ in range(self.measurement_protocol.candidate_warmup_runs)
            ]
            executions = [
                self.checked_execution(action, hint)
                for _ in range(self.measurement_protocol.candidate_measurement_runs)
            ]
        except QueryTimeout as error:
            result = {
                "candidate_id": candidate_id,
                "action": action,
                "action_valid": True,
                "constraints_satisfied": True,
                "compiled_hint": hint,
                "duplicate_of": None,
                "plan_sha256": fingerprint,
                "plain_explain": plain.document,
                "compact_plan": compact_plan(plain.document["Plan"]),
                "provisional_measurements": [],
                "provisional_speedup": 0.1,
                "execution_timed_out": True,
                "timeout_ms": error.timeout_ms,
                "errors_or_diagnostics": [str(error)],
                "pg_hint_plan": pg_hint_plan,
                "attempts_remaining": self.max_candidates - len(self.candidates) - 1,
            }
            self.candidates.append(result)
            return result
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        observations = [measured(execution) for execution in executions]
        median_ms = statistics.median(
            observation["execution_time_ms"] for observation in observations
        )
        result = {
            "candidate_id": candidate_id,
            "action": action,
            "action_valid": True,
            "constraints_satisfied": True,
            "compiled_hint": hint,
            "duplicate_of": None,
            "plan_sha256": fingerprint,
            "plain_explain": plain.document,
            "warmup": measured(warmups[-1]) if warmups else None,
            "measured_explain_analyze": executions[-1].document,
            "compact_plan": compact_plan(executions[-1].document["Plan"]),
            "provisional_measurements": observations,
            "provisional_median_execution_time_ms": median_ms,
            "provisional_speedup": score(
                self.default["median_execution_time_ms"], median_ms
            ),
            "execution_timed_out": False,
            "timeout_ms": self.timeout_ms,
            "errors_or_diagnostics": [],
            "pg_hint_plan": pg_hint_plan,
            "attempts_remaining": self.max_candidates - len(self.candidates) - 1,
        }
        self.candidates.append(result)
        self.by_fingerprint[fingerprint] = {
            "candidate_id": candidate_id,
            "measurements": observations,
            "median_execution_time_ms": median_ms,
        }
        return result

    def keep_default(self) -> dict[str, str]:
        if not self.default:
            raise RuntimeError("rollout baseline has not been started")
        if self.candidates:
            raise RuntimeError(
                "keep_default must be selected before submitting a candidate"
            )
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        self.kept_default = True
        return {"status": "kept_default"}

    def checked_execution(self, action: dict[str, Any], hint: str) -> ExplainResult:
        result = self.worker.explain(self.sql, self.timeout_ms, analyze=True, hint=hint)
        verification: Verification = verify_action(
            action, result.document["Plan"], result.hint_diagnostics
        )
        if not verification.valid:
            raise WorkerError("; ".join(verification.errors))
        return result

    def finish(self, rng: random.Random) -> dict[str, Any]:
        if self.kept_default:
            median_ms = self.default["median_execution_time_ms"]
            return {
                "measurement_protocol_id": self.measurement_protocol.protocol_id,
                "status": "completed",
                "decision": "keep_default",
                "winning_candidate_id": "default",
                "winning_plan_sha256": self.default["plan_sha256"],
                "score_source": "explicit_keep_default",
                "pair_orders": [],
                "candidate_measurements": self.default["measurements"],
                "default_measurements": self.default["measurements"],
                "candidate_median_execution_time_ms": median_ms,
                "default_median_execution_time_ms": median_ms,
                "score": 1.0,
                "trajectory_reward": 0.0,
                "invalid_attempt_count": 0,
                "duplicate_attempt_count": 0,
                "timeout_attempt_count": 0,
            }

        valid = [
            candidate
            for candidate in self.candidates
            if candidate["action_valid"]
            and candidate["constraints_satisfied"]
            and not candidate.get("execution_timed_out", False)
        ]
        invalid_count = len(self.candidates) - len(valid)
        timeout_count = sum(
            candidate.get("execution_timed_out", False) for candidate in self.candidates
        )
        duplicate_count = sum(
            candidate.get("duplicate_of") is not None for candidate in valid
        )
        if not valid:
            return {
                "measurement_protocol_id": (self.measurement_protocol.protocol_id),
                "status": "no_valid_candidate",
                "winning_candidate_id": None,
                "score": 0.0,
                "trajectory_reward": -3.0,
                "invalid_attempt_count": invalid_count,
                "duplicate_attempt_count": duplicate_count,
                "timeout_attempt_count": timeout_count,
            }

        winner = min(
            valid,
            key=lambda candidate: candidate["provisional_median_execution_time_ms"],
        )
        if winner["plan_sha256"] == self.default["plan_sha256"]:
            median_ms = self.default["median_execution_time_ms"]
            return {
                "measurement_protocol_id": self.measurement_protocol.protocol_id,
                "status": "completed",
                "decision": "candidate",
                "winning_candidate_id": winner["candidate_id"],
                "winning_plan_sha256": winner["plan_sha256"],
                "score_source": "default_fingerprint",
                "pair_orders": [],
                "candidate_measurements": self.default["measurements"],
                "default_measurements": self.default["measurements"],
                "candidate_median_execution_time_ms": median_ms,
                "default_median_execution_time_ms": median_ms,
                "score": 1.0,
                "trajectory_reward": (-0.10 * invalid_count - 0.05 * duplicate_count),
                "invalid_attempt_count": invalid_count,
                "duplicate_attempt_count": duplicate_count,
                "timeout_attempt_count": timeout_count,
            }

        action = winner["action"]
        hint = winner["compiled_hint"]
        for _ in range(self.measurement_protocol.final_warmup_pairs):
            try:
                self.checked_execution(action, hint)
            except QueryTimeout:
                return self.final_timeout(
                    winner, invalid_count, duplicate_count, timeout_count
                )
            self.worker.explain(self.sql, self.timeout_ms, analyze=True)

        pair_orders: list[list[str]] = []
        candidate_measurements: list[dict[str, Any]] = []
        default_measurements: list[dict[str, Any]] = []
        for _ in range(self.measurement_protocol.final_randomized_pairs):
            order = ["candidate", "default"]
            rng.shuffle(order)
            pair_orders.append(order)
            for label in order:
                if label == "candidate":
                    try:
                        candidate_measurements.append(
                            measured(self.checked_execution(action, hint))
                        )
                    except QueryTimeout:
                        return self.final_timeout(
                            winner,
                            invalid_count,
                            duplicate_count,
                            timeout_count,
                            pair_orders,
                            candidate_measurements,
                            default_measurements,
                        )
                else:
                    default_measurements.append(
                        measured(
                            self.worker.explain(self.sql, self.timeout_ms, analyze=True)
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
            "measurement_protocol_id": self.measurement_protocol.protocol_id,
            "status": "completed",
            "decision": "candidate",
            "winning_candidate_id": winner["candidate_id"],
            "winning_plan_sha256": winner["plan_sha256"],
            "score_source": "interleaved_measurement",
            "pair_orders": pair_orders,
            "candidate_measurements": candidate_measurements,
            "default_measurements": default_measurements,
            "candidate_median_execution_time_ms": candidate_median,
            "default_median_execution_time_ms": default_median,
            "score": final_score,
            "trajectory_reward": (
                math.log(final_score) - 0.10 * invalid_count - 0.05 * duplicate_count
            ),
            "invalid_attempt_count": invalid_count,
            "duplicate_attempt_count": duplicate_count,
            "timeout_attempt_count": timeout_count,
        }

    def final_timeout(
        self,
        winner: dict[str, Any],
        invalid_count: int,
        duplicate_count: int,
        timeout_count: int,
        pair_orders: list[list[str]] | None = None,
        candidate_measurements: list[dict[str, Any]] | None = None,
        default_measurements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        final_score = 0.1
        return {
            "measurement_protocol_id": self.measurement_protocol.protocol_id,
            "status": "candidate_timeout",
            "winning_candidate_id": winner["candidate_id"],
            "winning_plan_sha256": winner["plan_sha256"],
            "pair_orders": pair_orders or [],
            "candidate_measurements": candidate_measurements or [],
            "default_measurements": default_measurements or [],
            "timeout_ms": self.timeout_ms,
            "score": final_score,
            "trajectory_reward": (
                math.log(final_score) - 0.10 * invalid_count - 0.05 * duplicate_count
            ),
            "invalid_attempt_count": invalid_count,
            "duplicate_attempt_count": duplicate_count,
            "timeout_attempt_count": timeout_count + 1,
        }


class TrainingRolloutEvaluatorV1(RolloutEvaluator):
    """Cheaper reward measurement used only while training the policy."""

    def __init__(
        self,
        worker: PostgresWorker,
        task_set: TaskSet,
        task: dict[str, Any],
        *,
        global_timeout_ms: int = GLOBAL_TIMEOUT_MS,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        super().__init__(
            worker,
            task_set,
            task,
            global_timeout_ms=global_timeout_ms,
            measurement_protocol=RL_TRAINING_PROTOCOL_V1,
            max_candidates=max_candidates,
        )


class TrainingRolloutEvaluatorV2(RolloutEvaluator):
    """Training measurement with pinned task-relative execution timeouts."""

    def __init__(
        self,
        worker: PostgresWorker,
        task_set: TaskSet,
        task: dict[str, Any],
        calibrated_timeout: TaskTimeout,
        timeout_manifest_id: str,
        *,
        global_timeout_ms: int = GLOBAL_TIMEOUT_MS,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        super().__init__(
            worker,
            task_set,
            task,
            global_timeout_ms=global_timeout_ms,
            measurement_protocol=RL_TRAINING_PROTOCOL_V2,
            calibrated_timeout=calibrated_timeout,
            timeout_manifest_id=timeout_manifest_id,
            max_candidates=max_candidates,
        )
