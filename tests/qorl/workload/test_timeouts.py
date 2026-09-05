from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.workload.taskset import TaskSet
from qorl.workload.timeouts import CalibratedTimeouts

MANIFEST = Path("experiments/004-rl-run-v2/timeouts.json")


class TestCalibratedTimeout:
    def test_checked_manifest_covers_the_selected_400_tasks(
        self, repository_root: Path
    ) -> None:
        timeouts = CalibratedTimeouts.load(
            repository_root,
            MANIFEST,
            TaskSet.load(repository_root, "ceb-v1"),
        )

        assert len(timeouts.by_task_id) == 400
        assert len(timeouts.manifest_sha256) == 64
        assert timeouts.identity()["id"] == "qorl-rl-run-v2-timeouts-v1"
        replacement = timeouts.task("ceb-7a-7a14")
        assert replacement.timeout_ms == 5_000
        assert replacement.calibrated_default_ms == 1362.7295

    def test_legacy_manifest_runtime_image_can_be_enforced(
        self, repository_root: Path
    ) -> None:
        task_set = TaskSet.load(repository_root, "ceb-v1")
        with pytest.raises(RuntimeError, match="different runtime"):
            CalibratedTimeouts.load(
                repository_root,
                MANIFEST,
                task_set,
                {
                    "postgres_image_id": "sha256:different-runtime",
                    "postgres_config_id": "000-pgconf-default",
                },
            )

    def test_current_manifest_enforces_runtime_separately_from_data(
        self, repository_root: Path, tmp_path: Path
    ) -> None:
        task_set = TaskSet.load(repository_root, "ceb-v1")
        document = json.loads((repository_root / MANIFEST).read_text(encoding="utf-8"))
        document["data_identity"] = task_set.data_identity
        document["runtime_identity"] = {
            "postgres_image_id": "sha256:current-runtime",
            "postgres_config_id": "000-pgconf-default",
        }
        document.pop("database")
        path = tmp_path / "timeouts.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        loaded = CalibratedTimeouts.load(
            repository_root, path, task_set, document["runtime_identity"]
        )
        assert len(loaded.by_task_id) == 400
        with pytest.raises(RuntimeError, match="different runtime"):
            CalibratedTimeouts.load(
                repository_root,
                path,
                task_set,
                {
                    "postgres_image_id": "sha256:different-runtime",
                    "postgres_config_id": "000-pgconf-default",
                },
            )
