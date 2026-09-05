from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qorl.workload.taskset import TaskSet, TaskSetError


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_inventory_loads_without_a_database_archive(
    repository_root: Path, tmp_path: Path, workload: str
) -> None:
    source = repository_root / "benchmarks" / workload
    target = tmp_path / "benchmarks" / workload
    target.mkdir(parents=True)
    shutil.copyfile(source / "tasks.json", target / "tasks.json")
    shutil.copyfile(source / "manifest.json", target / "manifest.json")

    tasks = TaskSet.load(tmp_path, workload)

    assert tasks.data_identity == {"fixture_id": "imdb"}
    assert tasks.tasks
    assert tasks.task_set_id == workload
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_loads_checked_in_sql(repository_root: Path, workload: str) -> None:
    tasks = TaskSet.load(repository_root, workload)
    assert (
        tasks.load_sql(tasks.tasks[0].model_dump())
        .lstrip()
        .upper()
        .startswith("SELECT")
    )


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


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_checked_in_inventory_is_a_plain_task_list(
    repository_root: Path, workload: str
) -> None:
    directory = repository_root / "benchmarks" / workload
    records = json.loads((directory / "tasks.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    task_set = TaskSet.load(repository_root, workload)

    assert isinstance(records, list)
    assert records == [task.model_dump() for task in task_set.tasks]
    assert set(manifest) == {
        "schema_version",
        "workload_id",
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
        ("wrong_workload", "different workload"),
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
    elif change == "wrong_workload":
        manifest["workload_id"] = "ceb"
    payload = {"tasks": records} if change == "wrapper" else records
    (target / "tasks.json").write_text(json.dumps(payload))
    (target / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(TaskSetError, match=error):
        TaskSet.load(tmp_path, "job")
