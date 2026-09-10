from dataclasses import replace

import pytest

from qorl.taskset.exceptions import TaskSetError
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection, TemplateSample
from qorl.taskset.selection import parse_selection, select_tasks, validate_splits
from qorl.taskset.taskset import TaskSet


@pytest.mark.parametrize(
    ("expression", "role", "benchmark", "templates"),
    [
        ("test=job", TaskRole.TEST, BenchmarkId.JOB, ()),
        ("validation=ceb", TaskRole.VALIDATION, BenchmarkId.CEB, ()),
        (
            "train=ceb[1a:20,2a:30]",
            TaskRole.TRAIN,
            BenchmarkId.CEB,
            (TemplateSample("ceb-1a", 20), TemplateSample("ceb-2a", 30)),
        ),
        (
            " test = job[ 01:2, 02:1 ] ",
            TaskRole.TEST,
            BenchmarkId.JOB,
            (TemplateSample("job-01", 2), TemplateSample("job-02", 1)),
        ),
    ],
)
def test_parses_named_selection(
    expression: str,
    role: TaskRole,
    benchmark: BenchmarkId,
    templates: tuple[TemplateSample, ...],
) -> None:
    parsed = parse_selection(expression)
    assert parsed.role == role
    assert parsed.benchmark_id == benchmark
    assert parsed.templates == templates


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "ceb",
        "test=job,ceb",
        "train=ceb[]",
        "train=ceb[1a]",
        "train=ceb[1a:1,]",
        "train=ceb[1a:-1]",
        "train=ceb[1a:1.0]",
        "train=ceb[1a:10%]",
        "train=ceb[1a-2b:3]",
        "train=ceb[[1a:1]]",
        "train=ceb[1a:1]trailing",
    ],
)
def test_rejects_unsupported_grammar(expression: str) -> None:
    with pytest.raises(TaskSetError, match="invalid"):
        parse_selection(expression)


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("dev=ceb", "unknown selection role: dev"),
        ("test=imdb", "unknown benchmark: imdb"),
        ("train=ceb[1a:0]", "sample count must be positive"),
        ("train=ceb[1a:2,1a:3]", "duplicate template selection: ceb-1a"),
    ],
)
def test_reports_invalid_selection(expression: str, error: str) -> None:
    with pytest.raises(TaskSetError, match=error):
        parse_selection(expression)


def test_samples_without_replacement_and_round_trips(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    expressions = ["train=ceb[1a:20,2a:20]", "validation=ceb[4a:10]", "test=job"]
    selected = select_tasks(expressions, benchmark_task_sets, seed=42)
    assert selected == select_tasks(expressions, benchmark_task_sets, seed=42)
    assert len(selected[TaskRole.TRAIN].task_ids) == 40
    assert len(set(selected[TaskRole.TRAIN].task_ids)) == 40
    assert len(selected[TaskRole.VALIDATION].task_ids) == 10
    assert len(selected[TaskRole.TEST].task_ids) == 113
    for selection in selected.values():
        restored = TaskSelection.model_validate_json(selection.model_dump_json())
        task_set = benchmark_task_sets[restored.benchmark_id.value]
        assert [
            task.task_id for task in task_set.resolve(restored)
        ] == selection.task_ids
    assert (
        selected[TaskRole.TRAIN]
        != select_tasks(expressions, benchmark_task_sets, seed=43)[TaskRole.TRAIN]
    )


def test_sampling_is_independent_of_catalog_and_expression_order(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    expected = select_tasks(
        ["train=ceb[1a:20,2a:20]", "test=job"], benchmark_task_sets, seed=42
    )
    reordered = {
        benchmark: replace(task_set, tasks=list(reversed(task_set.tasks)))
        for benchmark, task_set in benchmark_task_sets.items()
    }
    assert (
        select_tasks(["test=job", "train=ceb[2a:20,1a:20]"], reordered, 42) == expected
    )
    only_first_template = select_tasks(["train=ceb[1a:20]"], reordered, 42)
    tasks = benchmark_task_sets["ceb"].resolve(expected[TaskRole.TRAIN])
    assert only_first_template[TaskRole.TRAIN].task_ids == [
        task.task_id for task in tasks if task.template_id == "ceb-1a"
    ]


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
def test_whole_benchmark_selects_every_task(
    benchmark_task_sets: dict[str, TaskSet], benchmark: str
) -> None:
    selection = select_tasks([f"test={benchmark}"], benchmark_task_sets, 42)[
        TaskRole.TEST
    ]
    assert selection.task_ids == sorted(
        task.task_id for task in benchmark_task_sets[benchmark].tasks
    )


@pytest.mark.parametrize(
    ("expressions", "error"),
    [
        ([], "at least one selection"),
        (["train=ceb[missing:1]"], "unknown template: ceb-missing"),
        (["test=job", "test=ceb"], "duplicate selection role: test"),
    ],
)
def test_rejects_invalid_requests(
    benchmark_task_sets: dict[str, TaskSet], expressions: list[str], error: str
) -> None:
    with pytest.raises(TaskSetError, match=error):
        select_tasks(expressions, benchmark_task_sets, seed=42)


def test_rejects_excessive_counts(benchmark_task_sets: dict[str, TaskSet]) -> None:
    available = benchmark_task_sets["ceb"].tasks_metadata["ceb-2a"].query_count
    with pytest.raises(
        TaskSetError, match=f"requested {available + 1} queries, but only {available}"
    ):
        select_tasks([f"train=ceb[2a:{available + 1}]"], benchmark_task_sets, seed=42)


def test_excludes_before_sampling_with_full_counts_and_stable_other_templates(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    expressions = ["train=ceb[2a:20]", "validation=ceb[4a:10]"]
    original = select_tasks(expressions, benchmark_task_sets, 42)
    excluded = original[TaskRole.TRAIN]
    selected = select_tasks(expressions, benchmark_task_sets, 42, exclude=[excluded])
    assert len(selected[TaskRole.TRAIN].task_ids) == 20
    assert not set(selected[TaskRole.TRAIN].task_ids) & set(excluded.task_ids)
    assert selected[TaskRole.VALIDATION] == original[TaskRole.VALIDATION]
    reordered = {
        key: replace(value, tasks=list(reversed(value.tasks)))
        for key, value in benchmark_task_sets.items()
    }
    assert selected == select_tasks(expressions, reordered, 42, exclude=[excluded])


def test_excludes_identical_sql_under_a_different_task_id(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    catalog = benchmark_task_sets["ceb"]
    first, second, *remaining = [t for t in catalog.tasks if t.template_id == "ceb-2a"]
    alias = second.model_copy(update={"sql_sha256": first.sql_sha256})
    modified = replace(catalog, tasks=[first, alias, *remaining])
    exclusion = TaskSelection(benchmark_id=BenchmarkId.CEB, task_ids=[first.task_id])
    selected = select_tasks(["test=ceb"], {"ceb": modified}, 42, exclude=[exclusion])
    assert selected[TaskRole.TEST].task_ids == sorted(t.task_id for t in remaining)


def test_multiple_exclusions_are_unioned_and_counts_use_remaining_queries(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    tasks = [
        t.task_id for t in benchmark_task_sets["ceb"].tasks if t.template_id == "ceb-2a"
    ]
    exclusions = [
        TaskSelection(benchmark_id=BenchmarkId.CEB, task_ids=tasks[:-1]),
        TaskSelection(benchmark_id=BenchmarkId.CEB, task_ids=tasks[:1]),
    ]
    selected = select_tasks(
        ["test=ceb[2a:1]"], benchmark_task_sets, 42, exclude=exclusions
    )
    assert selected[TaskRole.TEST].task_ids == tasks[-1:]
    with pytest.raises(TaskSetError, match="only 1 are available after exclusions"):
        select_tasks(["test=ceb[2a:2]"], benchmark_task_sets, 42, exclude=exclusions)


def test_rejects_unknown_excluded_tasks(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    excluded = TaskSelection(benchmark_id=BenchmarkId.CEB, task_ids=["ceb-missing"])
    with pytest.raises(TaskSetError, match="unknown selected task"):
        select_tasks(["test=ceb"], benchmark_task_sets, 42, exclude=[excluded])


def test_rejects_unused_excluded_benchmark(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    excluded = select_tasks(["test=job[01:1]"], benchmark_task_sets, 42)[TaskRole.TEST]
    with pytest.raises(TaskSetError, match="excluded benchmark job is not selected"):
        select_tasks(["test=ceb"], benchmark_task_sets, 42, exclude=[excluded])


def test_rejects_fully_excluded_benchmark(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    excluded = select_tasks(["test=job"], benchmark_task_sets, 42)[TaskRole.TEST]
    with pytest.raises(TaskSetError, match="no queries remain after exclusions"):
        select_tasks(["test=job"], benchmark_task_sets, 42, exclude=[excluded])


@pytest.mark.parametrize(
    ("train_template", "other_template"),
    [("1a", "2c"), ("2a", "2b"), ("3a", "3b"), ("9a", "9b"), ("1a", "1a")],
)
@pytest.mark.parametrize("role", ["validation", "test"])
def test_rejects_shared_topologies_between_roles(
    benchmark_task_sets: dict[str, TaskSet],
    train_template: str,
    other_template: str,
    role: str,
) -> None:
    with pytest.raises(
        TaskSetError,
        match=f"train ceb-{train_template} and {role} ceb-{other_template}",
    ):
        select_tasks(
            [f"train=ceb[{train_template}:1]", f"{role}=ceb[{other_template}:1]"],
            benchmark_task_sets,
            seed=42,
        )


def test_allows_shared_topologies_within_one_role(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    selected = select_tasks(["train=ceb[2a:2,2b:2]"], benchmark_task_sets, seed=42)
    assert len(selected[TaskRole.TRAIN].task_ids) == 4


def test_validates_imported_ids_and_topologies(
    benchmark_task_sets: dict[str, TaskSet],
) -> None:
    source = select_tasks(["train=ceb[2a:2]"], benchmark_task_sets, seed=42)
    validation = select_tasks(["validation=ceb[2b:2]"], benchmark_task_sets, seed=99)
    with pytest.raises(TaskSetError, match="topology overlap"):
        validate_splits(source | validation, benchmark_task_sets)
    missing = TaskSelection(benchmark_id=BenchmarkId.CEB, task_ids=["ceb-missing"])
    with pytest.raises(TaskSetError, match="unknown selected task"):
        validate_splits({TaskRole.TRAIN: missing}, benchmark_task_sets)


def test_rejects_missing_catalog() -> None:
    with pytest.raises(TaskSetError, match="benchmark catalog not loaded: job"):
        select_tasks(["test=job"], {}, seed=42)


def test_selection_does_not_read_sql(
    benchmark_task_sets: dict[str, TaskSet], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_sql(*_args: object) -> str:
        raise AssertionError("selection must not read SQL")

    monkeypatch.setattr(TaskSet, "load_sql", unexpected_sql)
    selected = select_tasks(["test=job"], benchmark_task_sets, seed=42)
    validate_splits(selected, benchmark_task_sets)
