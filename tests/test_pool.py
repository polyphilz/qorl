from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qorl.pool import (
    WorkerPool,
    WorkerResources,
    WorkerSlot,
    memory_bytes,
    validate_host_topology,
    worker_resources,
)


class FakeWorker:
    def close(self) -> None:
        pass


class WorkerPoolTest(unittest.TestCase):
    def test_resources_are_distinct_and_parameterized(self) -> None:
        resources = worker_resources(
            {
                "QORL_RL_WORKER_CPUSETS": "0-1;2-3;4-5;6-7",
                "QORL_RL_WORKER_MEMORY_LIMIT": "7g",
                "QORL_RL_WORKER_PORT_BASE": "56000",
            }
        )

        self.assertEqual([item.index for item in resources], [0, 1, 2, 3])
        self.assertEqual(
            resources[0].compose_environment["QORL_POSTGRES_CPUSET"],
            "0-1",
        )
        self.assertEqual(
            resources[3].compose_environment["QORL_POSTGRES_PORT"],
            "56003",
        )
        self.assertEqual(resources[0].memory_bytes, 7 * 1024**3)

    def test_claim_returns_workers_to_the_pool(self) -> None:
        slots = tuple(
            WorkerSlot(
                WorkerResources(
                    index,
                    str(index),
                    "1g",
                    memory_bytes("1g"),
                    56000 + index,
                ),
                FakeWorker(),  # type: ignore[arg-type]
            )
            for index in range(4)
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

            validate_host_topology(worker_resources({}), root)

            with self.assertRaisesRegex(RuntimeError, "four physical cores"):
                validate_host_topology(
                    worker_resources(
                        {
                            "QORL_RL_WORKER_CPUSETS": (
                                "0-1;2-3;4-5;6-7"
                            )
                        }
                    ),
                    root,
                )


if __name__ == "__main__":
    unittest.main()
