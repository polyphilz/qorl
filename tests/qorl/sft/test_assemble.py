from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from qorl.sft.assemble import action_families, select_tasks
from qorl.workload.taskset import TaskSet


class TestProtocolDataset:
    def test_selection_is_balanced_disjoint_and_deterministic(
        self, repository_root: Path
    ) -> None:
        seed = json.loads(
            (
                repository_root / "experiments/001-protocol-sft-v1/dataset.json"
            ).read_text()
        )["seed"]
        tasks = TaskSet.load(repository_root, "ceb-v1").inventory["tasks"]
        train = select_tasks(tasks, "train", 256, seed)
        validation = select_tasks(tasks, "validation", 64, seed)

        assert train == select_tasks(tasks, "train", 256, seed)
        assert len(train) == 256
        assert len(validation) == 64
        assert all(task["partition"] == "train" for task in train)
        assert all(task["partition"] == "validation" for task in validation)
        assert not {task["task_id"] for task in train} & {
            task["task_id"] for task in validation
        }
        assert not {task["template_id"] for task in train} & {
            task["template_id"] for task in validation
        }
        for selected in (train, validation):
            counts = Counter(task["template_id"] for task in selected)
            assert max(counts.values()) - min(counts.values()) <= 1

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
        assert action_families(action) == [
            "index_selection",
            "join",
            "leading",
            "memoize",
            "scan",
            "setting",
        ]
