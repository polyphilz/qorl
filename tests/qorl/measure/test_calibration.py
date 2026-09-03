from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.measure.calibration import buffers_stable, observation, selected_tasks
from qorl.workload.taskset import TaskSet


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


class TestCalibration:
    def test_ceb_calibration_resolves_the_400_task_training_selection(
        self, repository_root: Path
    ) -> None:
        task_set = TaskSet.load(repository_root, "ceb-v1")
        selection, split, tasks = selected_tasks(
            task_set,
            repository_root / "experiments/004-rl-run-v2/selection.json",
        )

        assert selection["inventory_id"] == "qorl-rl-run-v2"
        assert split == "train"
        assert len(tasks) == 400
        assert [task["task_id"] for task in tasks] == [
            item["task_id"] for item in selection["splits"]["train"]
        ]

    def test_multiple_selection_splits_require_an_explicit_name(
        self, repository_root: Path, tmp_path: Path
    ) -> None:
        task_set = TaskSet.load(repository_root, "ceb-v1")
        first, second = task_set.inventory["tasks"][:2]
        selection = {
            "inventory_id": "test-selection",
            "source": {"inventory_id": task_set.inventory["inventory_id"]},
            "splits": {
                "train": [{"task_id": first["task_id"]}],
                "validation": [{"task_id": second["task_id"]}],
            },
        }
        path = tmp_path / "selection.json"
        path.write_text(json.dumps(selection), encoding="utf-8")

        with pytest.raises(RuntimeError, match="pass --split"):
            selected_tasks(task_set, path)
        _, split, tasks = selected_tasks(task_set, path, "validation")

        assert split == "validation"
        assert [task["task_id"] for task in tasks] == [second["task_id"]]

    def test_buffer_stability_requires_same_plan_and_close_counts(self) -> None:
        first = observation(explain(hits=100, reads=5), 1)
        close = observation(explain(hits=101, reads=5), 2)
        far = observation(explain(hits=120, reads=5), 2)
        assert buffers_stable(first, close)
        assert not buffers_stable(first, far)
