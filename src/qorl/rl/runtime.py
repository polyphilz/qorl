from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path
from threading import BoundedSemaphore, Event

from qorl.postgres.config import PostgresConfig
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig, PoolManifest


class QorlRuntime(ContainerPool):
    def __init__(
        self,
        task_set: TaskSet,
        pool_config: PoolConfig,
        compose_project_name: str,
        postgres_config: PostgresConfig,
        max_concurrent_requests: int = 8,
    ) -> None:
        super().__init__(compose_project_name, pool_config, postgres_config)
        self.task_set = task_set
        self.requests = BoundedSemaphore(max_concurrent_requests)
        self.work: dict[asyncio.Task[None], Event] = {}

    def pool_manifest(self) -> PoolManifest:
        return self.manifest()


_runtime: QorlRuntime | None = None


def start(
    repository: Path,
    task_set: TaskSet,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    max_concurrent_requests: int = 8,
) -> QorlRuntime:
    global _runtime
    if _runtime is not None:
        raise RuntimeError("QORL runtime is already started")
    runtime = QorlRuntime(
        task_set,
        pool_config,
        f"qorl-rl-{os.getpid()}",
        postgres_config,
        max_concurrent_requests,
    )
    try:
        runtime.create()
        runtime.restore(repository / "data/imdb.tar.gz")
        runtime.start()
        runtime.load_indexes()
    except BaseException:
        runtime.close()
        raise
    _runtime = runtime
    return runtime


def current() -> QorlRuntime:
    if _runtime is None:
        raise RuntimeError("QORL runtime has not started")
    return _runtime


def stop() -> None:
    global _runtime
    runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.close()


async def drain() -> None:
    """All episode harnesses borrow this pool's requests and active worker registry."""
    if _runtime is None:
        return
    active = list(_runtime.work.items())
    for _, cancel in active:
        cancel.set()
    for work, _ in active:
        with suppress(Exception, asyncio.CancelledError):
            while not work.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(work)
            work.result()
