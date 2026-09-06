from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from qorl.measure.timeouts import CalibratedTimeouts
from qorl.postgres.config import PostgresConfig
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig

TIMEOUT_MANIFEST_ENV = "QORL_RL_TIMEOUT_MANIFEST"
POSTGRES_CONFIG_ENV = "QORL_RL_POSTGRES_CONFIG"
POOL_CONFIG_ENV = "QORL_RL_WORKER_POOL_CONFIG"


class QorlRuntime(ContainerPool):
    def __init__(
        self,
        repository: Path,
        task_set: TaskSet,
        pool_config: PoolConfig,
        project_name: str,
        postgres_config: PostgresConfig,
        calibrated_timeouts: CalibratedTimeouts | None = None,
    ) -> None:
        super().__init__(repository, project_name, pool_config, postgres_config)
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
    for name in (POSTGRES_CONFIG_ENV, POOL_CONFIG_ENV):
        if not environment.get(name, "").strip():
            raise RuntimeError(f"{name} must specify a configuration path")
    postgres_config = PostgresConfig.load(
        repository, Path(environment[POSTGRES_CONFIG_ENV])
    )
    pool_config = load_pool_config(repository, Path(environment[POOL_CONFIG_ENV]))
    runtime = QorlRuntime(
        repository,
        TaskSet.load(repository, "ceb"),
        pool_config,
        f"qorl-rl-{os.getpid()}",
        postgres_config,
    )
    try:
        runtime.create()
        runtime.restore(repository / "data/imdb.tar.gz")
        runtime.start()
        configured_timeouts = environment.get(TIMEOUT_MANIFEST_ENV)
        runtime.calibrated_timeouts = (
            CalibratedTimeouts.load(
                repository,
                Path(configured_timeouts),
                runtime.task_set,
                runtime.runtime_identity,
            )
            if configured_timeouts
            else None
        )
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
