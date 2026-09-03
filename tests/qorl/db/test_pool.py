from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.db.resources import (
    DEFAULT_TRAINING_PROFILE,
    load_runtime_profile,
    validate_host_topology,
)


ROOT = Path(__file__).resolve().parents[3]


class FakeWorker:
    def close(self) -> None:
        pass


class WorkerPoolTest(unittest.TestCase):
    def test_resources_are_distinct_and_parameterized(self) -> None:
        profile = load_runtime_profile(ROOT, DEFAULT_TRAINING_PROFILE)
        resources = profile.workers

        self.assertEqual([item.index for item in resources], [0, 1, 2, 3])
        self.assertEqual(
            resources[0].compose_environment["QORL_POSTGRES_CPUSET"],
            "0-3,16-19",
        )
        self.assertEqual(
            resources[3].compose_environment["QORL_POSTGRES_PORT"],
            "56003",
        )
        self.assertEqual(resources[0].memory_bytes, 8 * 1024**3)
        self.assertEqual(resources[0].physical_core_count, 4)
        self.assertEqual(resources[0].shm_bytes, 1024**3)

    def test_profile_rejects_overlapping_cpu_sets(self) -> None:
        with TemporaryDirectory() as temporary:
            repository = Path(temporary)
            profile_path = repository / "profile.json"
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

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                load_runtime_profile(repository, Path("profile.json"))

    def test_claim_returns_workers_to_the_pool(self) -> None:
        profile = load_runtime_profile(ROOT, DEFAULT_TRAINING_PROFILE)
        slots = tuple(
            WorkerSlot(resources, FakeWorker())  # type: ignore[arg-type]
            for resources in profile.workers
        )
        pool = WorkerPool(slots, "test-pool", "test-sha")  # type: ignore[arg-type]

        with pool.claim_worker() as first:
            with pool.claim_worker() as second:
                self.assertNotEqual(first.resources.index, second.resources.index)
        with pool.claim_worker() as next_slot:
            self.assertEqual(next_slot.resources.index, 2)

    def test_topology_requires_four_nonoverlapping_physical_cores(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cpu in range(32):
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0")
                (topology / "core_id").write_text(str(cpu % 16))

            profile = load_runtime_profile(ROOT, DEFAULT_TRAINING_PROFILE)
            validate_host_topology(profile.workers, root)

            with self.assertRaisesRegex(RuntimeError, "physical cores"):
                validate_host_topology(
                    (replace(profile.workers[0], cpuset="0-1"),),
                    root,
                )
