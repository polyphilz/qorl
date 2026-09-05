from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from qorl.sft import assemble
from qorl.sft.assemble import action_families, select_tasks
from qorl.workload.taskset import TaskSet


def test_finalize_without_cross_workload_report(
    repository_root: Path, tmp_path: Path, monkeypatch
) -> None:
    source = TaskSet.load(repository_root, "ceb").inventory
    tasks = [
        next(task for task in source["tasks"] if task["partition"] == partition)
        for partition in ("train", "validation")
    ]
    inventory = {**source, "tasks": tasks, "task_count": len(tasks)}
    ceb = tmp_path / "benchmarks/ceb"
    ceb.mkdir(parents=True)
    (ceb / "tasks.json").write_text(json.dumps(inventory))
    output = tmp_path / "dataset"
    for ordinal, task in enumerate(tasks):
        partition = task["partition"]
        directory = output / "demonstrations" / partition
        directory.mkdir(parents=True)
        document = {
            "metadata": {
                "ordinal": ordinal,
                "demonstration_id": f"demo-{ordinal}",
                "partition": partition,
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "data_identity": source["database"],
                "runtime_identity": {},
                "inspection_recipe": "test",
                "in_author_unique_plans_subset": ordinal == 0,
            },
            "messages": [],
            "tools": [],
        }
        (directory / "demo.json").write_text(json.dumps(document))
    monkeypatch.setattr(assemble, "SPLIT_COUNTS", {"train": 1, "validation": 1})
    monkeypatch.setattr(
        assemble,
        "validate_protocol_demo",
        lambda *_args: {
            "candidate_ids": [],
            "turn_count": 0,
            "canonical_sha256": "test",
        },
    )

    manifest = assemble.finalize_dataset(tmp_path, output, [], dataset_seed=0)

    assert manifest["schema_version"] == 2
    assert set(manifest["sources"]) == {"ceb_inventory"}
    assert "ineligible_ceb_templates" not in manifest["leakage"]
    assert manifest["counts"] == {"train": 1, "validation": 1}
    assert manifest["statistics"]["author_unique_plan_tasks"] == {"train": 1}
    assert not (ceb / "provenance/job-overlap.json").exists()


class TestProtocolDataset:
    def test_selection_is_balanced_disjoint_and_deterministic(
        self, repository_root: Path
    ) -> None:
        seed = json.loads(
            (
                repository_root / "experiments/001-protocol-sft-v1/dataset.json"
            ).read_text()
        )["seed"]
        tasks = TaskSet.load(repository_root, "ceb").inventory["tasks"]
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
