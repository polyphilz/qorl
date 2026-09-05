from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import (
    WorkerPool,
    WorkerSlot,
    start_pool,
)
from qorl.workload.taskset import TaskSet
from qorl.workload.timeouts import CalibratedTimeouts

TIMEOUT_MANIFEST_ENV = "QORL_RL_TIMEOUT_MANIFEST"


class QorlRuntime(WorkerPool):
    def __init__(
        self,
        task_set: TaskSet,
        workers: tuple[WorkerSlot, ...],
        pool_id: str,
        pool_config_sha256: str,
        pool_config_path: str | None = None,
        calibrated_timeouts: CalibratedTimeouts | None = None,
    ) -> None:
        super().__init__(workers, pool_id, pool_config_sha256, pool_config_path)
        self.task_set = task_set
        self.calibrated_timeouts = calibrated_timeouts
        self.data_identity = task_set.data_identity

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
        task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
        configured_timeouts = environment.get(TIMEOUT_MANIFEST_ENV)
        calibrated_timeouts = (
            CalibratedTimeouts.load(
                repository,
                Path(configured_timeouts),
                task_set,
                pool.runtime_identity,
            )
            if configured_timeouts
            else None
        )
        _runtime = QorlRuntime(
            task_set=task_set,
            workers=pool.workers,
            pool_id=pool.pool_id,
            pool_config_sha256=pool.pool_config_sha256,
            pool_config_path=pool.pool_config_path,
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
            slot.container.close()
