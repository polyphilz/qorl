from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from qorl.sft.assemble import action_families, select_tasks
from qorl.workload.taskset import TaskSet

ROOT = Path(__file__).resolve().parents[3]
DATASET_SEED = json.loads(
    (ROOT / "experiments/001-protocol-sft-v1/dataset.json").read_text()
)["seed"]


class ProtocolDatasetTest(unittest.TestCase):
    def test_selection_is_balanced_disjoint_and_deterministic(self) -> None:
        tasks = TaskSet.load(ROOT, "ceb-v1").inventory["tasks"]
        train = select_tasks(tasks, "train", 256, DATASET_SEED)
        validation = select_tasks(tasks, "validation", 64, DATASET_SEED)

        self.assertEqual(train, select_tasks(tasks, "train", 256, DATASET_SEED))
        self.assertEqual(len(train), 256)
        self.assertEqual(len(validation), 64)
        self.assertTrue(all(task["partition"] == "train" for task in train))
        self.assertTrue(all(task["partition"] == "validation" for task in validation))
        self.assertFalse(
            {task["task_id"] for task in train}
            & {task["task_id"] for task in validation}
        )
        self.assertFalse(
            {task["template_id"] for task in train}
            & {task["template_id"] for task in validation}
        )
        for selected in (train, validation):
            counts = Counter(task["template_id"] for task in selected)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_action_family_summary_includes_nested_features(self) -> None:
        action = {
            "version": 1,
            "leading": {"left": "a", "right": "b"},
            "joins": [
                {
                    "relations": ["a", "b"],
                    "force": "hash",
                    "memoize": "forbid",
                }
            ],
            "scans": [{"relation": "a", "force": "index", "indexes": ["i"]}],
            "settings": {"enable_hashjoin": True},
        }
        self.assertEqual(
            action_families(action),
            ["index_selection", "join", "leading", "memoize", "scan", "setting"],
        )
