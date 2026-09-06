from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qorl.taskset.exceptions import TaskSetError
from qorl.taskset.taskset import TaskSet


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
def test_inventory_loads_without_a_database_archive(
    repository_root: Path, tmp_path: Path, benchmark: str
) -> None:
    source = repository_root / "benchmarks" / benchmark
    target = tmp_path / "benchmarks" / benchmark
    target.mkdir(parents=True)
    shutil.copyfile(source / "tasks.json", target / "tasks.json")
    shutil.copyfile(source / "manifest.json", target / "manifest.json")

    tasks = TaskSet.load(tmp_path, benchmark)

    assert tasks.fixture_id == "imdb"
    assert tasks.inventory_path == target / "tasks.json"
    assert tasks.tasks
    assert tasks.task_set_id == benchmark
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
def test_loads_checked_in_sql(repository_root: Path, benchmark: str) -> None:
    tasks = TaskSet.load(repository_root, benchmark)
    task = tasks.tasks[0]
    sql = tasks.load_sql(task)
    assert sql == (tasks.inventory_path.parent / task.sql_path).read_text()
    assert sql.lstrip().upper().startswith("SELECT")


@pytest.mark.parametrize("sql_path", ["/outside.sql", "../outside.sql"])
def test_load_sql_rejects_unsafe_paths(repository_root: Path, sql_path: str) -> None:
    tasks = TaskSet.load(repository_root, "job")
    task = tasks.tasks[0].model_copy(update={"sql_path": sql_path})
    with pytest.raises(TaskSetError, match=f"invalid query path: {task.task_id}"):
        tasks.load_sql(task)


def test_load_sql_rejects_checksum_mismatch(repository_root: Path) -> None:
    tasks = TaskSet.load(repository_root, "job")
    task = tasks.tasks[0].model_copy(update={"sql_sha256": "0" * 64})
    with pytest.raises(TaskSetError, match=f"query checksum mismatch: {task.task_id}"):
        tasks.load_sql(task)


def test_inventory_requires_a_logical_fixture_id(
    repository_root: Path, tmp_path: Path
) -> None:
    source = json.loads((repository_root / "benchmarks/job/manifest.json").read_text())
    source.pop("fixture_id")
    target = tmp_path / "benchmarks/job"
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(json.dumps(source))
    shutil.copyfile(
        repository_root / "benchmarks/job/tasks.json", target / "tasks.json"
    )

    with pytest.raises(TaskSetError, match="fixture_id"):
        TaskSet.load(tmp_path, "job")


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
def test_checked_in_inventory_is_a_plain_task_list(
    repository_root: Path, benchmark: str
) -> None:
    directory = repository_root / "benchmarks" / benchmark
    records = json.loads((directory / "tasks.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    task_set = TaskSet.load(repository_root, benchmark)

    assert isinstance(records, list)
    assert records == [task.model_dump() for task in task_set.tasks]
    assert manifest["schema_version"] == 3
    assert set(manifest) == {
        "schema_version",
        "benchmark_id",
        "fixture_id",
        "description",
        "source",
    }
    expected_fields = {
        "task_id",
        "template_id",
        "sql_path",
        "sql_sha256",
        "tables",
        "relations",
        "join_edges",
        "table_count",
        "relation_count",
        "join_predicate_count",
    }
    assert all(set(record) == expected_fields for record in records)
    for task in task_set.tasks:
        assert task.table_count == len(task.tables)
        assert task.relation_count == len(task.relations)
        assert task.join_predicate_count == len(task.join_edges)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("wrapper", "valid array"),
        ("duplicate", "duplicate task IDs"),
        ("missing_id", "task_id"),
        ("invalid_relation", "alias"),
        ("wrong_benchmark", "different benchmark"),
    ],
)
def test_rejects_invalid_inventory(
    repository_root: Path, tmp_path: Path, change: str, error: str
) -> None:
    source = repository_root / "benchmarks/job"
    target = tmp_path / "benchmarks/job"
    target.mkdir(parents=True)
    records = json.loads((source / "tasks.json").read_text())[:1]
    manifest = json.loads((source / "manifest.json").read_text())
    if change == "duplicate":
        records.append(records[0])
    elif change == "missing_id":
        del records[0]["task_id"]
    elif change == "invalid_relation":
        records[0]["relations"][0]["alias"] = 1
    elif change == "wrong_benchmark":
        manifest["benchmark_id"] = "ceb"
    payload = {"tasks": records} if change == "wrapper" else records
    (target / "tasks.json").write_text(json.dumps(payload))
    (target / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(TaskSetError, match=error):
        TaskSet.load(tmp_path, "job")
