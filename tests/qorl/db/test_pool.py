from __future__ import annotations

from pathlib import Path

import pytest

from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot, load_pool
from qorl.db.resources import (
    DEFAULT_POOL_CONFIG,
    load_runtime_profile,
)
from qorl.db.worker import PostgresWorker


class TestWorkerPool:
    def test_claim_returns_workers_to_the_pool(
        self, repository_root: Path, database_fixture: DatabaseFixture
    ) -> None:
        profile = load_runtime_profile(repository_root, DEFAULT_POOL_CONFIG)
        slots = []
        for resources in profile.workers:
            container = PostgresContainer(
                database_fixture,
                f"test-pool-{resources.index}",
                profile,
                resources,
            )
            slots.append(WorkerSlot(resources, container, PostgresWorker(container)))
        pool = WorkerPool(tuple(slots), "test-pool", "test-sha")

        with pool.claim_worker() as first, pool.claim_worker() as second:
            assert first.resources.index != second.resources.index
        with pool.claim_worker() as next_slot:
            assert next_slot.resources.index == 2
        pool.close()


@pytest.mark.parametrize(
    ("config_id", "worker_count"),
    [("000-poolconf-1x32", 1), ("001-poolconf-2x16", 2), ("002-poolconf-4x8", 4)],
)
def test_pool_selection_accepts_each_worker_count(
    repository_root: Path, config_id: str, worker_count: int
) -> None:
    configured = Path("docker/worker_pool/configs") / config_id
    from_environment = load_pool(
        repository_root, {"QORL_RL_WORKER_POOL_CONFIG": str(configured)}
    )
    explicit = load_pool(
        repository_root,
        {"QORL_RL_WORKER_POOL_CONFIG": "missing-config"},
        pool_config_path=configured,
    )
    assert explicit == from_environment
    assert len(explicit.workers) == worker_count
    assert explicit.profile_id == config_id
