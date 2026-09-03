from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qorl.measure import run
from qorl.measure.run import TaskRun


class FakeContainer:
    def __init__(self) -> None:
        self.captures: list[tuple[Path, str]] = []
        self.closed = False

    def capture_environment(self, path: Path, phase: str) -> None:
        self.captures.append((path, phase))

    def close(self) -> None:
        self.closed = True


class FakePool:
    def __init__(self) -> None:
        self.containers = [FakeContainer(), FakeContainer()]
        self.workers = tuple(
            SimpleNamespace(
                resources=SimpleNamespace(index=index),
                container=container,
            )
            for index, container in enumerate(self.containers)
        )

    def manifest(self) -> dict[str, object]:
        return {"worker_count": len(self.workers)}

    def close(self) -> None:
        for container in self.containers:
            container.close()


def test_task_run_owns_pool_capture_manifest_and_loop(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pool = FakePool()
    monkeypatch.setattr(run, "start_pool", lambda *_: pool)
    manifest: dict[str, object] = {}
    task_run = TaskRun(
        SimpleNamespace(),
        "test-run",
        tmp_path,
        tmp_path / "report.json",
        manifest,
        pool_field="database_pool",
        environment_dir=tmp_path / "environment",
    )

    with task_run:
        completions = list(task_run.map([1, 2], lambda _, value: value * 2))

    assert sorted(item.result for item in completions if item.result is not None) == [
        2,
        4,
    ]
    assert manifest["database_pool"] == {"worker_count": 2}
    assert all(container.closed for container in pool.containers)
    assert pool.containers[0].captures == [
        (tmp_path / "environment/worker-0", "pre"),
        (tmp_path / "environment/worker-0", "post"),
    ]


def test_task_run_can_leave_capture_to_each_policy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pool = FakePool()
    monkeypatch.setattr(run, "start_pool", lambda *_: pool)
    task_run = TaskRun(
        SimpleNamespace(),
        "test-run",
        tmp_path,
        tmp_path / "report.json",
        {},
        pool_field="database_pool",
        capture_environment=False,
    )

    with task_run:
        pass

    assert all(not container.captures for container in pool.containers)
    assert all(container.closed for container in pool.containers)
