from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from qorl.fixture import DatabaseFixture, TaskSet
from qorl.pool import (
    WorkerPool,
    WorkerSlot,
    start_pool,
)
from qorl.timeouts import CalibratedTimeouts


TIMEOUT_MANIFEST_ENV = "QORL_RL_TIMEOUT_MANIFEST"


class QorlRuntime(WorkerPool):
    def __init__(
        self,
        task_set: TaskSet,
        workers: tuple[WorkerSlot, ...],
        pool_id: str,
        pool_config_sha256: str,
        calibrated_timeouts: CalibratedTimeouts | None = None,
    ) -> None:
        super().__init__(workers, pool_id, pool_config_sha256)
        self.task_set = task_set
        self.calibrated_timeouts = calibrated_timeouts

    def pool_manifest(self) -> dict[str, object]:
        return self.manifest()


_runtime: QorlRuntime | None = None


def start(
    repository: Path,
    environment: Mapping[str, str] = os.environ,
) -> QorlRuntime:
    global _runtime
    if _runtime is not None:
        raise RuntimeError("QORL runtime is already started")
    fixture = DatabaseFixture.load(repository)
    pool: WorkerPool | None = None
    try:
        pool = start_pool(fixture, f"qorl-rl-{os.getpid()}", environment)
        task_set = TaskSet.load(repository, "ceb-v1", fixture.identity)
        configured_timeouts = environment.get(TIMEOUT_MANIFEST_ENV)
        calibrated_timeouts = (
            CalibratedTimeouts.load(
                repository,
                Path(configured_timeouts),
                task_set,
            )
            if configured_timeouts
            else None
        )
        _runtime = QorlRuntime(
            task_set=task_set,
            workers=pool.workers,
            pool_id=pool.pool_id,
            pool_config_sha256=pool.pool_config_sha256,
            calibrated_timeouts=calibrated_timeouts,
        )
    except BaseException:
        if pool is not None:
            pool.close()
        raise
    return _runtime


def current() -> QorlRuntime:
    if _runtime is None:
        raise RuntimeError("QORL runtime has not started")
    return _runtime


def stop() -> None:
    global _runtime
    runtime, _runtime = _runtime, None
    if runtime is not None:
        for slot in reversed(runtime.workers):
            slot.worker.close()
