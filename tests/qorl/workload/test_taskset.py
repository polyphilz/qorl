from __future__ import annotations

from pathlib import Path

import pytest

from qorl.db.fixture import FixtureError
from qorl.workload.taskset import TaskSet


@pytest.fixture(scope="module")
def task_sets(repository_root: Path) -> tuple[TaskSet, TaskSet]:
    job = TaskSet.load(repository_root, "job")
    return job, TaskSet.load(repository_root, "ceb", job.data_identity)


class TestTaskSet:
    def test_job_and_ceb_require_the_same_database(
        self, task_sets: tuple[TaskSet, TaskSet]
    ) -> None:
        job, ceb = task_sets
        assert job.inventory["database"] == ceb.inventory["database"]

    def test_loads_checked_in_sql_from_both_task_sets(
        self, task_sets: tuple[TaskSet, TaskSet]
    ) -> None:
        for task_set in task_sets:
            sql = task_set.load_sql(task_set.inventory["tasks"][0])
            assert sql.lstrip().upper().startswith("SELECT")

    def test_rejects_a_different_database_identity(
        self, repository_root: Path, task_sets: tuple[TaskSet, TaskSet]
    ) -> None:
        job, _ceb = task_sets
        different = {**job.data_identity, "archive_sha256": "wrong"}
        with pytest.raises(FixtureError, match="requires a different database fixture"):
            TaskSet.load(repository_root, "ceb", different)

    def test_inventory_runtime_metadata_is_not_part_of_data_identity(
        self, repository_root: Path, task_sets: tuple[TaskSet, TaskSet]
    ) -> None:
        job, _ceb = task_sets
        expected = {
            **job.inventory["database"],
            "postgres_image_id": "sha256:different-runtime",
        }
        loaded = TaskSet.load(repository_root, "ceb", expected)

        assert loaded.data_identity == job.data_identity
