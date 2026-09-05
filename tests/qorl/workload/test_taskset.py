from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qorl.db.fixture import FixtureError
from qorl.workload.taskset import TaskSet


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_inventory_loads_without_a_database_archive(
    repository_root: Path, tmp_path: Path, workload: str
) -> None:
    source = repository_root / "benchmarks" / workload
    target = tmp_path / "benchmarks" / workload
    target.mkdir(parents=True)
    shutil.copyfile(source / "tasks.json", target / "tasks.json")

    tasks = TaskSet.load(tmp_path, workload)

    assert tasks.data_identity == {"fixture_id": "imdb"}
    assert "database" not in tasks.inventory
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_loads_checked_in_sql(repository_root: Path, workload: str) -> None:
    tasks = TaskSet.load(repository_root, workload)
    assert (
        tasks.load_sql(tasks.inventory["tasks"][0])
        .lstrip()
        .upper()
        .startswith("SELECT")
    )


def test_inventory_requires_a_logical_fixture_id(
    repository_root: Path, tmp_path: Path
) -> None:
    source = json.loads((repository_root / "benchmarks/job/tasks.json").read_text())
    source.pop("fixture_id")
    target = tmp_path / "benchmarks/job"
    target.mkdir(parents=True)
    (target / "tasks.json").write_text(json.dumps(source))

    with pytest.raises(FixtureError, match="fixture ID"):
        TaskSet.load(tmp_path, "job")
