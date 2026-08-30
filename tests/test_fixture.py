from __future__ import annotations

import unittest
from pathlib import Path

from qorl.fixture import FixtureError, TaskSet


ROOT = Path(__file__).resolve().parents[1]


class TaskSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.job = TaskSet.load(ROOT, "job-v1")
        cls.ceb = TaskSet.load(ROOT, "ceb-v1", cls.job.inventory["database"])

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
        with self.assertRaisesRegex(
            FixtureError, "requires a different database snapshot"
        ):
            TaskSet.load(ROOT, "ceb-v1", {"snapshot_id": "wrong"})


if __name__ == "__main__":
    unittest.main()
