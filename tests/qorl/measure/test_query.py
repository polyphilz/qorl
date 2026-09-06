import pytest

from qorl.measure.query import buffers_stable, measure_query, observation
from qorl.postgres.schemas import ExplainResult


def explain(
    *, hits: int = 100, reads: int = 5, table: str = "title", elapsed_ms: float = 1.5
) -> ExplainResult:
    return ExplainResult(
        document={
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": table,
                "Plan Rows": 100,
                "Shared Hit Blocks": hits,
                "Shared Read Blocks": reads,
            },
            "Planning Time": 0.2,
            "Execution Time": elapsed_ms,
        },
        hint_diagnostics="",
    )


@pytest.mark.parametrize(
    ("hits", "reads", "table", "stable"),
    [
        (101, 5, "title", True),
        (120, 5, "title", False),
        (100, 6, "title", False),
        (100, 5, "name", False),
    ],
)
def test_buffer_stability_requires_same_plan_and_close_counts(
    hits: int, reads: int, table: str, stable: bool
) -> None:
    first = observation(explain(), 1)
    second = observation(explain(hits=hits, reads=reads, table=table), 2)
    assert buffers_stable(first, second) is stable


@pytest.mark.parametrize("num_trials", [1, 2, 20])
def test_stable_warmups_stop_at_two_and_trials_are_separate(num_trials: int) -> None:
    executions: list[ExplainResult] = []

    def execute() -> ExplainResult:
        result = explain(elapsed_ms=float(len(executions) + 1))
        executions.append(result)
        return result

    runs = list(measure_query(execute, max_warmup_runs=5, num_trials=num_trials))
    warmups = [run for run in runs if run.is_warmup]
    trials = [run for run in runs if not run.is_warmup]
    assert len(executions) == 2 + num_trials
    assert [run.observation.run for run in warmups] == [1, 2]
    assert [run.observation.run for run in trials] == list(range(1, num_trials + 1))
    assert [run.observation.execution_time_ms for run in trials] == list(
        range(3, num_trials + 3)
    )
    assert trials[0].explain is executions[2]


@pytest.mark.parametrize("max_warmup_runs", [2, 3, 5, 7])
def test_unstable_warmups_respect_the_configured_cap(max_warmup_runs: int) -> None:
    executions = 0

    def execute() -> ExplainResult:
        nonlocal executions
        executions += 1
        return explain(hits=100 * executions)

    runs = list(measure_query(execute, max_warmup_runs=max_warmup_runs, num_trials=2))
    assert sum(run.is_warmup for run in runs) == max_warmup_runs
    assert executions == max_warmup_runs + 2


def test_warmups_can_stabilize_between_minimum_and_cap() -> None:
    hits = iter([100, 150, 151, 152])

    def execute() -> ExplainResult:
        return explain(hits=next(hits))

    runs = list(measure_query(execute, max_warmup_runs=5, num_trials=1))
    assert [run.is_warmup for run in runs] == [True, True, True, False]


@pytest.mark.parametrize(("max_warmup_runs", "num_trials"), [(1, 2), (2, 0)])
def test_invalid_counts_do_not_execute(max_warmup_runs: int, num_trials: int) -> None:
    def execute() -> ExplainResult:
        pytest.fail("invalid counts must be rejected before executing SQL")

    with pytest.raises(ValueError, match="must be at least"):
        list(
            measure_query(
                execute, max_warmup_runs=max_warmup_runs, num_trials=num_trials
            )
        )


def test_query_failure_stops_execution() -> None:
    executions = 0

    def execute() -> ExplainResult:
        nonlocal executions
        executions += 1
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        list(measure_query(execute, max_warmup_runs=5, num_trials=20))
    assert executions == 1
