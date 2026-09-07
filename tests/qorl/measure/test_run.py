from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from qorl.measure import run
from qorl.measure.run import TaskRun
from qorl.postgres.config import PostgresConfig
from qorl.worker_pool.schemas import PoolConfig, PoolManifest


class FakePool:
    def __init__(
        self, postgres_config: PostgresConfig, pool_config: PoolConfig
    ) -> None:
        self.workers = tuple(
            SimpleNamespace(resources=resources)
            for resources in pool_config.workers[:2]
        )
        self.record = PoolManifest(
            id=pool_config.profile_id,
            path=str(pool_config.path),
            config_sha256=pool_config.sha256,
            worker_count=len(self.workers),
            workers=[slot.resources.manifest() for slot in self.workers],
            postgres_config=postgres_config.manifest(),
        )
        self.captures: list[tuple[int, Path, str]] = []
        self.closed = False

    def manifest(self) -> PoolManifest:
        return self.record

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
    pool = FakePool(postgres_config, pool_config)
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
    assert manifest["database_pool"] == pool.manifest().model_dump()
    assert pool.closed
    assert [(path, phase) for index, path, phase in pool.captures if index == 0] == [
        (tmp_path / "environment/worker-0", "pre"),
        (tmp_path / "environment/worker-0", "post"),
    ]


def test_task_run_skips_post_capture_and_closes_pool_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = FakePool(postgres_config, pool_config)
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
    )

    with pytest.raises(ValueError, match="task failed"), task_run:
        raise ValueError("task failed")

    assert [(index, phase) for index, _, phase in pool.captures] == [
        (0, "pre"),
        (1, "pre"),
    ]
    assert pool.closed


def test_task_run_cancels_pending_tasks_after_an_unhandled_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = FakePool(postgres_config, pool_config)
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
    )

    with task_run, pytest.raises(ValueError, match="unexpected task failure"):
        list(task_run.map([1, 2, 3], lambda _, value: value))

    assert all(future.cancelled() for future in ImmediateExecutor.futures[1:])
