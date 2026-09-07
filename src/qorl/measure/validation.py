"""Validate candidate plans with plain EXPLAIN, without collecting latency samples."""

from concurrent.futures import CancelledError
from threading import Event

from pydantic import JsonValue

from qorl.measure.protocols import QueryExecutor, SqlSource
from qorl.measure.schemas import (
    Baseline,
    Candidate,
    MeasurementStatus,
    ToolResultStatus,
)
from qorl.plans.catalog import TaskCatalog
from qorl.plans.exceptions import ActionError
from qorl.plans.fingerprint import (
    PLAN_FINGERPRINT_VERSION,
    plan_sha256,
    structural_plan_sha256,
    timing_reuse_key,
)
from qorl.plans.schemas import PlanAction
from qorl.plans.verify import compact_plan, hint_status, verify_action
from qorl.postgres.exceptions import QueryTimeoutError
from qorl.taskset.schemas import Task


def validate_candidate(
    worker: QueryExecutor,
    sql: str,
    catalog: TaskCatalog,
    raw_action: JsonValue,
    *,
    candidate_id: str,
    timeout_ms: int,
    attempts_remaining: int,
) -> Candidate:
    """Parse, compile and verify one attempt; infrastructure errors propagate.

    A malformed action consumes its issued attempt without querying PostgreSQL.
    A planning timeout is candidate evidence, not a measured execution time.
    Novelty and execution-duplicate classification belong to the owning rollout.
    """
    candidate = Candidate(
        candidate_id=candidate_id,
        action=raw_action,
        action_valid=False,
        constraints_satisfied=False,
        compiled_hint="",
        duplicate_of=None,
        plan_sha256=None,
        plan_fingerprint_version=PLAN_FINGERPRINT_VERSION,
        structural_plan_sha256=None,
        structural_duplicate_of=None,
        timing_reuse_key=None,
        execution_timed_out=False,
        measurement_status=MeasurementStatus.NOT_MEASURED,
        pg_hint_plan=None,
        attempts_remaining=attempts_remaining,
    )
    try:
        action = PlanAction.from_raw(raw_action, catalog)
    except ActionError as error:
        return candidate.model_copy(update={"errors_or_diagnostics": [str(error)]})
    hint = action.compile()
    candidate = candidate.model_copy(
        update={"action": action.to_wire(), "action_valid": True, "compiled_hint": hint}
    )
    try:
        plain = worker.explain(sql, timeout_ms, hint=hint)
    except QueryTimeoutError as error:
        return candidate.model_copy(
            update={
                "execution_timed_out": True,
                "timeout_ms": error.timeout_ms,
                "errors_or_diagnostics": [str(error)],
            }
        )
    plan = plain.document["Plan"]
    verification = verify_action(action.to_wire(), plan, plain.hint_diagnostics)
    fingerprint = plan_sha256(plan)
    return candidate.model_copy(
        update={
            "constraints_satisfied": verification.valid,
            "plan_sha256": fingerprint,
            "structural_plan_sha256": structural_plan_sha256(plan),
            "timing_reuse_key": timing_reuse_key(fingerprint, action),
            "plain_explain": plain.document,
            "compact_plan": compact_plan(plan),
            "pg_hint_plan": hint_status(plain.hint_diagnostics),
            "errors_or_diagnostics": list(verification.errors),
        }
    )


class PlanValidationEvaluator[ExecutorT: QueryExecutor]:
    """Own attempt IDs and plan evidence, independently of the timing evaluator."""

    def __init__(
        self,
        worker: ExecutorT,
        task_set: SqlSource,
        task: Task,
        *,
        default_timeout_ms: int,
        max_candidates: int,
        cancel: Event | None = None,
    ) -> None:
        if default_timeout_ms < 1 or max_candidates < 1:
            raise ValueError("timeout and candidate limit must be positive")
        self._worker = worker
        self.task = task
        self.sql = task_set.load_sql(task)
        self.catalog = TaskCatalog.from_postgres(task, worker.indexes)
        self.timeout_ms = default_timeout_ms
        self.max_candidates = max_candidates
        self.default: Baseline | None = None
        self.candidates: list[Candidate] = []
        self.kept_default = False
        self.by_structure: dict[str, str] = {}
        self.by_reuse_key: dict[str, str] = {}
        self.cancel = cancel

    def check_cancelled(self) -> None:
        """Stop between bounded calls without abandoning an in-flight database worker."""
        if self.cancel is not None and self.cancel.is_set():
            raise CancelledError("rollout cancelled")

    @property
    def worker(self) -> ExecutorT:
        return self._worker

    def start(self) -> Baseline:
        """Explain the default once, recording identities but no execution samples."""
        if self.default is not None:
            raise RuntimeError("rollout baseline has already been started")
        self.check_cancelled()
        plain = self.worker.explain(self.sql, self.timeout_ms)
        plan = plain.document["Plan"]
        fingerprint = plan_sha256(plan)
        structure = structural_plan_sha256(plan)
        reuse_key = timing_reuse_key(fingerprint)
        self.default = Baseline(
            plan_sha256=fingerprint,
            structural_plan_sha256=structure,
            timing_reuse_key=reuse_key,
            plan_fingerprint_version=PLAN_FINGERPRINT_VERSION,
            plain_explain=plain.document,
            compact_plan=compact_plan(plan),
            median_execution_time_ms=None,
        )
        self.by_structure[structure] = "default"
        self.by_reuse_key[reuse_key] = "default"
        self.check_cancelled()
        return self.default

    def evaluate(self, raw_action: JsonValue) -> Candidate:
        """Issue one attempt and distinguish structural from execution duplicates."""
        self.check_cancelled()
        if self.default is None:
            raise RuntimeError("rollout baseline has not been started")
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        if len(self.candidates) >= self.max_candidates:
            raise RuntimeError("rollout candidate budget is exhausted")
        candidate = validate_candidate(
            self.worker,
            self.sql,
            self.catalog,
            raw_action,
            candidate_id=f"candidate-{len(self.candidates) + 1:02d}",
            timeout_ms=self.timeout_ms,
            attempts_remaining=self.max_candidates - len(self.candidates) - 1,
        )
        if candidate.constraints_satisfied:
            structure = candidate.structural_plan_sha256
            reuse_key = candidate.timing_reuse_key
            if structure is None or reuse_key is None:
                raise RuntimeError("validated plan has no identity")
            candidate = candidate.model_copy(
                update={
                    "structural_duplicate_of": self.by_structure.get(structure),
                    "duplicate_of": self.by_reuse_key.get(reuse_key),
                }
            )
            self.by_structure.setdefault(structure, candidate.candidate_id)
            self.by_reuse_key.setdefault(reuse_key, candidate.candidate_id)
        self.candidates.append(candidate)
        self.check_cancelled()
        return candidate

    def keep_default(self) -> dict[str, str]:
        """Record an explicit default decision before any candidate submission."""
        if self.default is None:
            raise RuntimeError("rollout baseline has not been started")
        if self.candidates:
            raise RuntimeError(
                "keep_default must be selected before submitting a candidate"
            )
        if self.kept_default:
            raise RuntimeError("rollout already kept the default plan")
        self.kept_default = True
        return {"status": ToolResultStatus.KEPT_DEFAULT.value}
