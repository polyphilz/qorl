from __future__ import annotations

import contextlib
import os
import queue
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from qorl.db.config import PostgresConfig
from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import (
    DEFAULT_POOL_CONFIG,
    RuntimeProfile,
    WorkerResources,
    load_runtime_profile,
    validate_host_topology,
)
from qorl.db.worker import PostgresWorker


@dataclass(frozen=True)
class WorkerSlot:
    resources: WorkerResources
    container: PostgresContainer
    worker: PostgresWorker


@dataclass
class WorkerPool:
    workers: tuple[WorkerSlot, ...]
    pool_id: str
    pool_config_sha256: str
    pool_config_path: str | None = None

    def __post_init__(self) -> None:
        self._available: queue.Queue[WorkerSlot] = queue.Queue()
        for slot in self.workers:
            self._available.put(slot)

    @contextlib.contextmanager
    def claim_worker(self) -> Iterator[WorkerSlot]:
        slot = self._available.get()
        try:
            yield slot
        finally:
            self._available.put(slot)

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.pool_id,
            "path": self.pool_config_path,
            "config_sha256": self.pool_config_sha256,
            "worker_count": len(self.workers),
            "workers": [slot.resources.manifest() for slot in self.workers],
            "postgres_config": self.postgres_config.manifest().model_dump(),
        }

    @property
    def postgres_config(self) -> PostgresConfig:
        if not self.workers:
            raise RuntimeError("PostgreSQL worker pool is empty")
        return self.workers[0].container.postgres_config

    @property
    def runtime_identity(self) -> dict[str, str]:
        if not self.workers:
            raise RuntimeError("PostgreSQL worker pool is empty")
        return self.workers[0].worker.fixture.runtime_identity_for(self.postgres_config)

    def close(self) -> None:
        for slot in reversed(self.workers):
            slot.container.close()


def load_pool(
    repository: Path,
    environment: Mapping[str, str] = os.environ,
    *,
    pool_config_path: Path | None = None,
) -> RuntimeProfile:
    configured = pool_config_path or Path(
        environment.get("QORL_RL_WORKER_POOL_CONFIG", DEFAULT_POOL_CONFIG)
    )
    return load_runtime_profile(repository, configured)


def start_pool(
    fixture: DatabaseFixture,
    project_name: str,
    environment: Mapping[str, str] = os.environ,
    *,
    postgres_config: PostgresConfig | None = None,
    pool_config: RuntimeProfile | None = None,
) -> WorkerPool:
    profile = pool_config or load_pool(fixture.repository, environment)
    selected_config = postgres_config or PostgresConfig.load(fixture.repository)
    validate_host_topology(profile.workers)
    slots: list[WorkerSlot] = []
    try:
        for resources in profile.workers:
            container = PostgresContainer(
                fixture,
                f"{project_name}-{resources.index}",
                profile,
                resources,
                selected_config,
            )
            worker = PostgresWorker(container)
            slots.append(WorkerSlot(resources, container, worker))
            container.start()
            worker.assert_fixture()
    except BaseException:
        for slot in reversed(slots):
            slot.container.close()
        raise
    return WorkerPool(
        tuple(slots), profile.profile_id, profile.sha256, str(profile.path)
    )
