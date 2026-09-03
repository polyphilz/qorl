from __future__ import annotations

import contextlib
import os
import queue
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import (
    DEFAULT_TRAINING_PROFILE,
    RuntimeProfile,
    WorkerResources,
    load_runtime_profile,
    validate_host_topology,
)
from qorl.db.worker import PostgresWorker

DEFAULT_POOL_CONFIG = DEFAULT_TRAINING_PROFILE
EXPECTED_POOL_WORKERS = 4


@dataclass(frozen=True)
class WorkerSlot:
    resources: WorkerResources
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
        }

    def close(self) -> None:
        for slot in reversed(self.workers):
            slot.worker.close()


def load_pool(
    repository: Path, environment: Mapping[str, str] = os.environ
) -> RuntimeProfile:
    configured = Path(
        environment.get("QORL_RL_WORKER_POOL_CONFIG", DEFAULT_POOL_CONFIG)
    )
    profile = load_runtime_profile(repository, configured)
    if len(profile.workers) != EXPECTED_POOL_WORKERS:
        raise ValueError("the training profile must define exactly four workers")
    return profile


def start_pool(
    fixture: DatabaseFixture,
    project_name: str,
    environment: Mapping[str, str] = os.environ,
) -> WorkerPool:
    profile = load_pool(fixture.repository, environment)
    validate_host_topology(profile.workers)
    slots: list[WorkerSlot] = []
    try:
        for resources in profile.workers:
            worker = PostgresWorker(
                fixture,
                f"{project_name}-{resources.index}",
                runtime_profile=profile,
                resources=resources,
            )
            slots.append(WorkerSlot(resources, worker))
            worker.start()
    except BaseException:
        for slot in reversed(slots):
            slot.worker.close()
        raise
    return WorkerPool(
        tuple(slots), profile.profile_id, profile.sha256, str(profile.path)
    )
