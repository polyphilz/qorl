from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl.measure import calibration, run
from qorl.postgres.client import PostgresClient
from qorl.postgres.schemas import ExplainResult, PostgresIndexes
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool import containers as pool_module
from qorl.worker_pool.containers import ContainerPool


@pytest.mark.parametrize(
    ("config_id", "worker_count"),
    [("000-poolconf-1x32", 1), ("001-poolconf-2x16", 2), ("002-poolconf-4x8", 4)],
)
@pytest.mark.parametrize(("max_warmup_runs", "num_trials"), [(5, 20), (2, 2), (3, 4)])
def test_calibration_starts_and_records_the_selected_pool(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_id: str,
    worker_count: int,
    max_warmup_runs: int,
    num_trials: int,
    postgres_indexes: PostgresIndexes,
) -> None:
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")
    task_set = TaskSet.load(repository_root, "job")
    task_set = replace(
        task_set,
        tasks=task_set.tasks[:5],
    )
    started: list[ContainerPool] = []
    executions: list[str] = []
    benchmarks: list[str] = []

    def load(cls: type[TaskSet], repository: Path, benchmark: str) -> TaskSet:
        benchmarks.append(benchmark)
        return task_set

    def execute(
        worker: PostgresClient,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        assert analyze
        assert timeout_ms == 120_000
        assert hint == ""
        executions.append(sql)
        return ExplainResult(
            document={
                "Plan": {
                    "Node Type": "Seq Scan",
                    "Relation Name": "title",
                    "Plan Rows": 100,
                    "Actual Rows": 10,
                    "Actual Loops": 1,
                    "Shared Hit Blocks": 100,
                    "Shared Read Blocks": 5,
                },
                "Planning Time": 0.2,
                "Execution Time": 1.5,
            },
            hint_diagnostics="",
        )

    monkeypatch.setattr(TaskSet, "load", classmethod(load))
    monkeypatch.setattr(pool_module, "validate_host_topology", lambda resources: None)
    monkeypatch.setattr(
        ContainerPool, "start", lambda container: started.append(container)
    )
    monkeypatch.setattr(run, "capture_environment", lambda *args: None)
    monkeypatch.setattr(ContainerPool, "create", lambda container: None)
    monkeypatch.setattr(ContainerPool, "restore", lambda *args: None)
    monkeypatch.setattr(PostgresClient, "explain", execute)
    index_reads = Mock(return_value=postgres_indexes)
    monkeypatch.setattr(PostgresClient, "read_indexes", index_reads)

    output = calibration.calibrate(
        tmp_path,
        postgres_config_path=repository_root
        / "docker/postgres/configs/000-pgconf-default",
        pool_config_path=repository_root / "docker/worker_pool/configs" / config_id,
        max_warmup_runs=max_warmup_runs,
        num_trials=num_trials,
    )
    manifest = json.loads((output / "calibration.json").read_text())
    assert len(started) == 1
    index_reads.assert_called_once()
    assert len(started[0].workers) == worker_count
    assert benchmarks == ["job"]
    assert len(executions) == 5 * (2 + num_trials)
    assert manifest["status"] == "completed"
    assert manifest["completed_task_count"] == 5
    assert manifest["runtime_identity"] == {"postgres_config_id": "000-pgconf-default"}
    assert manifest["protocol"]["worker_count"] == worker_count
    assert manifest["protocol"]["concurrent_tasks"] == worker_count
    assert manifest["protocol"]["minimum_warmup_runs"] == 2
    assert manifest["protocol"]["maximum_warmup_runs"] == max_warmup_runs
    assert manifest["protocol"]["measurement_runs"] == num_trials
    assert manifest["worker_pool"]["id"] == config_id
    assert manifest["worker_pool"]["worker_count"] == worker_count
    assert len(manifest["worker_pool"]["workers"]) == worker_count
    assert manifest["worker_pool"]["config_sha256"] == started[0].pool_config.sha256
    assert config_id in output.name
    assert "000-pgconf-default" in output.name
    for task in task_set.tasks:
        record = json.loads((output / "tasks" / f"{task.task_id}.json").read_text())
        assert len(record["warmups"]) == 2
        assert len(record["measurements"]) == num_trials
        assert [item["run"] for item in record["measurements"]] == list(
            range(1, num_trials + 1)
        )
        assert record["summary"]["measurement_count"] == num_trials
        assert record["summary"]["median_execution_time_ms"] == 1.5
        assert record["summary"]["coefficient_of_variation"] == 0
        assert record["summary"]["distinct_plan_count"] == 1
        first = record["measurements"][0]
        assert first == {
            "run": 1,
            "execution_time_ms": 1.5,
            "planning_time_ms": 0.2,
            "shared_hit_blocks": 100,
            "shared_read_blocks": 5,
            "plan_sha256": record["summary"]["plan_sha256s"][0],
        }
        assert record["representative_explain_analyze"]["Execution Time"] == 1.5


@pytest.mark.parametrize(
    ("max_warmup_runs", "num_trials", "message"),
    [
        (1, 20, "max_warmup_runs must be at least 2"),
        (5, 1, "num_trials must be at least 2"),
    ],
)
def test_calibration_validates_counts_before_loading_configs(
    tmp_path: Path, max_warmup_runs: int, num_trials: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calibration.calibrate(
            tmp_path,
            postgres_config_path=tmp_path / "missing-pgconf",
            pool_config_path=tmp_path / "missing-poolconf",
            max_warmup_runs=max_warmup_runs,
            num_trials=num_trials,
        )
