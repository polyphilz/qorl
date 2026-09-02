from __future__ import annotations

import unittest
from pathlib import Path

from qorl.db.fixture import DatabaseFixture, FixtureError
from qorl.workload.taskset import TaskSet


ROOT = Path(__file__).resolve().parents[1]


class TaskSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.job = TaskSet.load(ROOT, "job-v1")
        cls.ceb = TaskSet.load(ROOT, "ceb-v1", cls.job.data_identity)

    def test_job_and_ceb_require_the_same_database(self) -> None:
        self.assertEqual(
            self.job.inventory["database"],
            self.ceb.inventory["database"],
        )

    def test_loads_checked_in_sql_from_both_task_sets(self) -> None:
        for task_set in (self.job, self.ceb):
            sql = task_set.load_sql(task_set.inventory["tasks"][0])
            self.assertTrue(sql.lstrip().upper().startswith("SELECT"))

    def test_rejects_a_different_database_identity(self) -> None:
        different = {**self.job.data_identity, "snapshot_id": "wrong"}
        with self.assertRaisesRegex(
            FixtureError, "requires a different database snapshot"
        ):
            TaskSet.load(ROOT, "ceb-v1", different)

    def test_inventory_runtime_metadata_is_not_part_of_data_identity(self) -> None:
        expected = {
            **self.job.inventory["database"],
            "postgres_image_id": "sha256:different-runtime",
        }
        loaded = TaskSet.load(ROOT, "ceb-v1", expected)

        self.assertEqual(loaded.data_identity, self.job.data_identity)

    def test_fixture_splits_data_and_runtime_identity(self) -> None:
        fixture = DatabaseFixture(
            repository=ROOT,
            snapshot_manifest_path=ROOT / "snapshot.json",
            archive_path=ROOT / "snapshot.tar.gz",
            snapshot={
                "fixture_id": "job-v1",
                "snapshot_id": "snapshot",
                "archive": {"sha256": "archive"},
                "postgresql": {"system_identifier": "system"},
                "image": {
                    "id": "sha256:image",
                    "benchmark_config_id": "benchmark-v2",
                },
            },
        )

        self.assertEqual(
            fixture.data_identity,
            {
                "fixture_id": "job-v1",
                "snapshot_id": "snapshot",
                "snapshot_archive_sha256": "archive",
                "postgres_system_identifier": "system",
            },
        )
        self.assertEqual(
            fixture.runtime_identity,
            {
                "postgres_image_id": "sha256:image",
                "benchmark_config_id": "benchmark-v2",
            },
        )


if __name__ == "__main__":
    unittest.main()
