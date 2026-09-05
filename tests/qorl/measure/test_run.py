from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from qorl.measure import run
from qorl.measure.run import TaskRun
from qorl.postgres.config import PostgresConfig
from qorl.worker_pool.schemas import PoolConfig


class FakePool:
    def __init__(self) -> None:
        self.workers = tuple(
            SimpleNamespace(resources=SimpleNamespace(index=index))
            for index in range(2)
        )
        self.captures: list[tuple[int, Path, str]] = []
        self.closed = False

    def manifest(self) -> dict[str, object]:
        return {"worker_count": len(self.workers)}

    def close(self) -> None:
        self.closed = True


class ImmediateExecutor:
    futures: ClassVar[list[Future[int]]] = []

    def __init__(self, *, max_workers: int) -> None:
        del max_workers
        self.futures = []
        ImmediateExecutor.futures = self.futures

    def __enter__(self) -> ImmediateExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def submit(self, function: object, pool: object, item: int) -> Future[int]:
        del function, pool
        future: Future[int] = Future()
        if item == 1:
            future.set_exception(ValueError("unexpected task failure"))
        self.futures.append(future)
        return future


def test_task_run_owns_pool_capture_manifest_and_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = FakePool()
    monkeypatch.setattr(run, "start_pool", lambda *_, **__: pool)
    monkeypatch.setattr(
        run,
        "capture_environment",
        lambda pool, slot, path, phase: pool.captures.append(
            (slot.resources.index, path, phase)
        ),
    )
    manifest: dict[str, object] = {}
    task_run = TaskRun(
        tmp_path,
        "test-run",
        tmp_path,
        tmp_path / "report.json",
        manifest,
        pool_field="database_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
        environment_dir=tmp_path / "environment",
    )

    with task_run:
        completions = list(task_run.map([1, 2], lambda _, value: value * 2))

    assert sorted(item.result for item in completions if item.result is not None) == [
        2,
        4,
    ]
    assert manifest["database_pool"] == {"worker_count": 2}
    assert pool.closed
    assert [(path, phase) for index, path, phase in pool.captures if index == 0] == [
        (tmp_path / "environment/worker-0", "pre"),
        (tmp_path / "environment/worker-0", "post"),
    ]


def test_task_run_can_leave_capture_to_each_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = FakePool()
    monkeypatch.setattr(run, "start_pool", lambda *_, **__: pool)
    monkeypatch.setattr(
        run,
        "capture_environment",
        lambda pool, slot, path, phase: pool.captures.append(
            (slot.resources.index, path, phase)
        ),
    )
    task_run = TaskRun(
        tmp_path,
        "test-run",
        tmp_path,
        tmp_path / "report.json",
        {},
        pool_field="database_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
        capture_environment=False,
    )

    with task_run:
        pass

    assert not pool.captures
    assert pool.closed


def test_task_run_cancels_pending_tasks_after_an_unhandled_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = FakePool()
    monkeypatch.setattr(run, "start_pool", lambda *_, **__: pool)
    monkeypatch.setattr(
        run,
        "capture_environment",
        lambda pool, slot, path, phase: pool.captures.append(
            (slot.resources.index, path, phase)
        ),
    )
    monkeypatch.setattr(run, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(run, "as_completed", lambda futures: iter(futures))
    task_run = TaskRun(
        tmp_path,
        "test-run",
        tmp_path,
        tmp_path / "report.json",
        {},
        pool_field="database_pool",
        postgres_config=postgres_config,
        pool_config=pool_config,
        capture_environment=False,
    )

    with task_run, pytest.raises(ValueError, match="unexpected task failure"):
        list(task_run.map([1, 2, 3], lambda _, value: value))

    assert all(future.cancelled() for future in ImmediateExecutor.futures[1:])
