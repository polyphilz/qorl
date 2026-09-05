from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qorl.db import pool as pool_module
from qorl.db.config import DEFAULT_POSTGRES_CONFIG
from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.worker import PostgresWorker
from qorl.measure import calibration
from qorl.measure.calibration import buffers_stable, observation, selected_tasks
from qorl.workload.taskset import TaskSet


@pytest.mark.parametrize(
    ("config_id", "worker_count"),
    [("000-poolconf-1x32", 1), ("001-poolconf-2x16", 2), ("002-poolconf-4x8", 4)],
)
def test_calibration_starts_and_records_the_selected_pool(
    repository_root: Path,
    database_fixture: DatabaseFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_id: str,
    worker_count: int,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(database_fixture.snapshot))
    fixture = replace(database_fixture, snapshot_manifest_path=snapshot_path)
    task_set = TaskSet.load(repository_root, "job-v1", fixture.data_identity)
    task_set = replace(
        task_set,
        inventory={
            **task_set.inventory,
            "tasks": task_set.inventory["tasks"][:5],
            "task_count": 5,
        },
    )
    started: list[PostgresContainer] = []
    executions: list[str] = []

    def execute(
        worker: PostgresWorker, sql: str, timeout_ms: int, *, hint: str | None = None
    ) -> dict:
        executions.append(sql)
        return explain()

    monkeypatch.setattr(
        DatabaseFixture, "load", classmethod(lambda cls, repository: fixture)
    )
    monkeypatch.setattr(TaskSet, "load", classmethod(lambda cls, *args: task_set))
    monkeypatch.setattr(pool_module, "validate_host_topology", lambda resources: None)
    monkeypatch.setattr(
        PostgresContainer, "start", lambda container: started.append(container)
    )
    monkeypatch.setattr(PostgresContainer, "capture_environment", lambda *args: None)
    monkeypatch.setattr(PostgresWorker, "assert_snapshot", lambda worker: None)
    monkeypatch.setattr(PostgresWorker, "explain_analyze", execute)

    output = calibration.calibrate(
        tmp_path,
        postgres_config_path=repository_root / DEFAULT_POSTGRES_CONFIG,
        pool_config_path=repository_root / "docker/worker_pool/configs" / config_id,
    )
    manifest = json.loads((output / "calibration.json").read_text())
    assert len(started) == worker_count
    assert len(executions) == 5 * (2 + 20)
    assert manifest["status"] == "completed"
    assert manifest["completed_task_count"] == 5
    assert manifest["protocol"]["worker_count"] == worker_count
    assert manifest["protocol"]["concurrent_tasks"] == worker_count
    assert manifest["worker_pool"]["id"] == config_id
    assert manifest["worker_pool"]["worker_count"] == worker_count
    assert len(manifest["worker_pool"]["workers"]) == worker_count
    assert manifest["worker_pool"]["config_sha256"] == started[0].runtime_profile.sha256
    assert config_id in output.name
    assert "000-pgconf-default" in output.name


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
