from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qorl.calibration import (
    PLAN_FINGERPRINT_VERSION,
    buffers_stable,
    canonical_plan,
    observation,
    plan_sha256,
    selected_tasks,
)
from qorl.fixture import TaskSet


ROOT = Path(__file__).resolve().parents[1]


def explain(*, rows: int = 10, hits: int = 100, reads: int = 5) -> dict:
    return {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "title",
            "Plan Rows": 100,
            "Actual Rows": rows,
            "Actual Loops": 1,
            "Shared Hit Blocks": hits,
            "Shared Read Blocks": reads,
        },
        "Planning Time": 0.2,
        "Execution Time": 1.5,
    }


class CalibrationTest(unittest.TestCase):
    def test_ceb_calibration_resolves_the_400_task_training_selection(self) -> None:
        task_set = TaskSet.load(ROOT, "ceb-v1")
        selection, split, tasks = selected_tasks(
            task_set,
            ROOT / "data/ceb/ceb-v1/rl-run-v2.json",
        )

        self.assertEqual(selection["inventory_id"], "qorl-rl-run-v2")
        self.assertEqual(split, "train")
        self.assertEqual(len(tasks), 400)
        self.assertEqual(
            [task["task_id"] for task in tasks],
            [item["task_id"] for item in selection["splits"]["train"]],
        )

    def test_multiple_selection_splits_require_an_explicit_name(self) -> None:
        task_set = TaskSet.load(ROOT, "ceb-v1")
        first, second = task_set.inventory["tasks"][:2]
        selection = {
            "inventory_id": "test-selection",
            "source": {"inventory_id": task_set.inventory["inventory_id"]},
            "splits": {
                "train": [{"task_id": first["task_id"]}],
                "validation": [{"task_id": second["task_id"]}],
            },
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            path.write_text(json.dumps(selection), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "pass --split"):
                selected_tasks(task_set, path)
            _, split, tasks = selected_tasks(task_set, path, "validation")

        self.assertEqual(split, "validation")
        self.assertEqual([task["task_id"] for task in tasks], [second["task_id"]])

    def test_plan_fingerprint_ignores_runtime_observations(self) -> None:
        first = explain(rows=10, hits=100, reads=5)["Plan"]
        first.update(
            {
                "Cache Hits": 10,
                "Disk Usage": 0,
                "Hash Batches": 1,
                "HashAgg Batches": 1,
                "Index Searches": 3,
                "Peak Memory Usage": 12,
                "Subplans Removed": 0,
                "Rows Removed by Join Filter": 4,
                "Sort Space Used": 8,
                "Workers Launched": 2,
            }
        )
        second = explain(rows=20, hits=200, reads=9)["Plan"]
        second.update(
            {
                "Cache Hits": 100,
                "Disk Usage": 8192,
                "Hash Batches": 2,
                "HashAgg Batches": 4,
                "Index Searches": 30,
                "Peak Memory Usage": 24,
                "Subplans Removed": 2,
                "Rows Removed by Join Filter": 40,
                "Sort Space Used": 16,
                "Workers Launched": 1,
            }
        )
        self.assertEqual(plan_sha256(first), plan_sha256(second))

    def test_plain_and_analyzed_hash_aggregate_have_the_same_fingerprint(self) -> None:
        plain = {
            "Node Type": "Aggregate",
            "Strategy": "Hashed",
            "Partial Mode": "Simple",
            "Parallel Aware": False,
            "Async Capable": False,
            "Startup Cost": 100.0,
            "Total Cost": 120.0,
            "Plan Rows": 10,
            "Plan Width": 16,
            "Group Key": ["title.kind_id"],
            "Planned Partitions": 0,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "title",
                    "Alias": "title",
                    "Startup Cost": 0.0,
                    "Total Cost": 90.0,
                    "Plan Rows": 1000,
                    "Plan Width": 8,
                }
            ],
        }
        analyzed = json.loads(json.dumps(plain))
        analyzed.update(
            {
                "Actual Startup Time": 0.1,
                "Actual Total Time": 1.2,
                "Actual Rows": 10,
                "Actual Loops": 1,
                "HashAgg Batches": 3,
                "Peak Memory Usage": 129,
                "Disk Usage": 456,
                "Shared Hit Blocks": 100,
                "Shared Read Blocks": 5,
            }
        )
        analyzed["Plans"][0].update(
            {
                "Actual Startup Time": 0.01,
                "Actual Total Time": 0.8,
                "Actual Rows": 1000,
                "Actual Loops": 1,
                "Rows Removed by Filter": 4,
                "Shared Hit Blocks": 100,
            }
        )

        self.assertEqual(PLAN_FINGERPRINT_VERSION, 3)
        self.assertEqual(canonical_plan(analyzed), plain)
        self.assertEqual(plan_sha256(analyzed), plan_sha256(plain))

    def test_incremental_sort_runtime_groups_do_not_change_fingerprint(self) -> None:
        plain = {
            "Node Type": "Incremental Sort",
            "Sort Key": ["title.kind_id", "title.id"],
            "Presorted Key": ["title.kind_id"],
            "Plan Rows": 1000,
        }
        analyzed = {
            **plain,
            "Full-sort Groups": {
                "Group Count": 4,
                "Sort Methods Used": ["quicksort"],
                "Sort Space Memory": {
                    "Average Sort Space Used": 27,
                    "Peak Sort Space Used": 27,
                },
            },
            "Pre-sorted Groups": {
                "Group Count": 2,
                "Sort Methods Used": ["external merge"],
                "Sort Space Disk": {
                    "Average Sort Space Used": 128,
                    "Peak Sort Space Used": 256,
                },
            },
        }
        self.assertEqual(plan_sha256(analyzed), plan_sha256(plain))

    def test_fingerprint_keeps_planner_fields_near_runtime_fields(self) -> None:
        plan = {
            "Node Type": "Gather",
            "Parallel Aware": False,
            "Workers Planned": 2,
            "Plan Rows": 100,
            "Plans": [
                {
                    "Node Type": "Aggregate",
                    "Strategy": "Hashed",
                    "Planned Partitions": 4,
                    "Plan Rows": 10,
                }
            ],
        }
        analyzed = {
            **plan,
            "Workers Launched": 1,
            "Workers": [{"Worker Number": 0, "Actual Rows": 50}],
        }

        self.assertEqual(canonical_plan(analyzed), plan)
        self.assertNotEqual(
            plan_sha256(plan),
            plan_sha256({**plan, "Workers Planned": 1}),
        )
        changed_partitions = json.loads(json.dumps(plan))
        changed_partitions["Plans"][0]["Planned Partitions"] = 2
        self.assertNotEqual(plan_sha256(plan), plan_sha256(changed_partitions))

    def test_plan_fingerprint_detects_physical_plan_change(self) -> None:
        first = explain()["Plan"]
        second = {**first, "Node Type": "Index Scan"}
        self.assertNotEqual(plan_sha256(first), plan_sha256(second))

    def test_buffer_stability_requires_same_plan_and_close_counts(self) -> None:
        first = observation(explain(hits=100, reads=5), 1)
        close = observation(explain(hits=101, reads=5), 2)
        far = observation(explain(hits=120, reads=5), 2)
        self.assertTrue(buffers_stable(first, close))
        self.assertFalse(buffers_stable(first, far))


if __name__ == "__main__":
    unittest.main()
