from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerSlot
from qorl.db.resources import (
    DEFAULT_POOL_CONFIG,
    load_runtime_profile,
    validate_host_topology,
)
from qorl.db.worker import PostgresWorker
from qorl.workload.taskset import TaskSet
from qorl_training.runtime import QorlRuntime

ROOT = Path(__file__).resolve().parents[2]


def database_fixture() -> DatabaseFixture:
    return DatabaseFixture(
        repository=ROOT,
        manifest_path=ROOT / "imdb/archive.json",
        archive_path=ROOT / "imdb/imdb.tar.gz",
        manifest=json.loads((ROOT / "imdb/archive.json").read_text(encoding="utf-8")),
    )


class WorkerPoolTest(unittest.TestCase):
    def test_resources_are_distinct_and_parameterized(self) -> None:
        resources = load_runtime_profile(ROOT, DEFAULT_POOL_CONFIG).workers

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

    def test_claim_returns_workers_to_the_pool(self) -> None:
        profile = load_runtime_profile(ROOT, DEFAULT_POOL_CONFIG)
        fixture = database_fixture()
        slots = []
        for resources in profile.workers:
            container = PostgresContainer(
                fixture,
                f"test-pool-{resources.index}",
                profile,
                resources,
            )
            slots.append(WorkerSlot(resources, container, PostgresWorker(container)))
        runtime = QorlRuntime(
            TaskSet.load(ROOT, "ceb", fixture.data_identity),
            tuple(slots),
            "test-pool",
            "test-sha",
        )
        self.addCleanup(runtime.close)

        with runtime.claim_worker() as first, runtime.claim_worker() as second:
            self.assertNotEqual(first.resources.index, second.resources.index)
        with runtime.claim_worker() as next_slot:
            self.assertEqual(next_slot.resources.index, 2)

    def test_topology_requires_four_nonoverlapping_physical_cores(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cpu in range(32):
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0")
                (topology / "core_id").write_text(str(cpu % 16))

            profile = load_runtime_profile(ROOT, DEFAULT_POOL_CONFIG)
            validate_host_topology(profile.workers, root)

            with self.assertRaisesRegex(RuntimeError, "physical cores"):
                validate_host_topology(
                    (replace(profile.workers[0], cpuset="0-1"),),
                    root,
                )


if __name__ == "__main__":
    unittest.main()
