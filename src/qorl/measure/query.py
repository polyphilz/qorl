"""Run one query's warmups and timed trials without constructing a report."""

from collections.abc import Callable, Iterator

from qorl.measure.schemas import QueryObservation, QueryRun
from qorl.plans.fingerprint import plan_sha256
from qorl.postgres.schemas import ExplainResult

MIN_WARMUP_RUNS = 2
BUFFER_STABILITY_TOLERANCE = 0.02


def observation(explain: ExplainResult, run_number: int) -> QueryObservation:
    """Extract timing, buffer counts, and plan identity from one execution."""
    document = explain.document
    plan = document["Plan"]
    return QueryObservation(
        run=run_number,
        execution_time_ms=document["Execution Time"],
        planning_time_ms=document["Planning Time"],
        shared_hit_blocks=plan.get("Shared Hit Blocks", 0),
        shared_read_blocks=plan.get("Shared Read Blocks", 0),
        plan_sha256=plan_sha256(plan),
    )


def buffers_stable(previous: QueryObservation, current: QueryObservation) -> bool:
    """Check consecutive plan identities and buffer counts, not physical disk I/O."""
    if previous.plan_sha256 != current.plan_sha256:
        return False
    for left, right in (
        (previous.shared_hit_blocks, current.shared_hit_blocks),
        (previous.shared_read_blocks, current.shared_read_blocks),
    ):
        scale = max(1, left, right)
        if abs(left - right) / scale > BUFFER_STABILITY_TOLERANCE:
            return False
    return True


def measure_query(
    execute: Callable[[], ExplainResult], *, max_warmup_runs: int, num_trials: int
) -> Iterator[QueryRun]:
    """Yield adaptive warmups, then trials; execute supplies SQL and its timeout."""
    if max_warmup_runs < MIN_WARMUP_RUNS:
        raise ValueError(f"max_warmup_runs must be at least {MIN_WARMUP_RUNS}")
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")

    previous: QueryObservation | None = None
    for run_number in range(1, max_warmup_runs + 1):
        explain = execute()
        current = observation(explain, run_number)
        yield QueryRun(is_warmup=True, observation=current, explain=explain)
        if previous is not None and buffers_stable(previous, current):
            break
        previous = current

    for run_number in range(1, num_trials + 1):
        explain = execute()
        yield QueryRun(
            is_warmup=False,
            observation=observation(explain, run_number),
            explain=explain,
        )
