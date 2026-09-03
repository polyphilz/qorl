from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.db.resources import (
    DEFAULT_TRAINING_PROFILE,
    load_runtime_profile,
    validate_host_topology,
)
from qorl.db.worker import PostgresWorker


class TestWorkerPool:
    def test_resources_are_distinct_and_parameterized(
        self, repository_root: Path
    ) -> None:
        profile = load_runtime_profile(repository_root, DEFAULT_TRAINING_PROFILE)
        resources = profile.workers

        assert [item.index for item in resources] == [0, 1, 2, 3]
        assert resources[0].compose_environment["QORL_POSTGRES_CPUSET"] == "0-3,16-19"
        assert resources[3].compose_environment["QORL_POSTGRES_PORT"] == "56003"
        assert resources[0].memory_bytes == 8 * 1024**3
        assert resources[0].physical_core_count == 4
        assert resources[0].shm_bytes == 1024**3

    def test_profile_rejects_overlapping_cpu_sets(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile_id": "overlap",
                    "worker_count": 2,
                    "cpuset_mems": "0",
                    "worker_memory_limit": "1g",
                    "worker_shm_size": "1g",
                    "worker_port_base": 56000,
                    "workers": [
                        {
                            "slot": 0,
                            "physical_core_count": 2,
                            "cpuset": "0-1",
                        },
                        {
                            "slot": 1,
                            "physical_core_count": 2,
                            "cpuset": "1-2",
                        },
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match="must not overlap"):
            load_runtime_profile(tmp_path, Path("profile.json"))

    def test_claim_returns_workers_to_the_pool(
        self, repository_root: Path, database_fixture: DatabaseFixture
    ) -> None:
        profile = load_runtime_profile(repository_root, DEFAULT_TRAINING_PROFILE)
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

    def test_topology_requires_four_nonoverlapping_physical_cores(
        self, repository_root: Path, tmp_path: Path
    ) -> None:
        for cpu in range(32):
            topology = tmp_path / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "physical_package_id").write_text("0")
            (topology / "core_id").write_text(str(cpu % 16))

        profile = load_runtime_profile(repository_root, DEFAULT_TRAINING_PROFILE)
        validate_host_topology(profile.workers, tmp_path)

        with pytest.raises(RuntimeError, match="physical cores"):
            validate_host_topology(
                (replace(profile.workers[0], cpuset="0-1"),),
                tmp_path,
            )
