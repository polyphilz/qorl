from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from qorl.db.pool import WorkerSlot
from qorl.db.resources import (
    DEFAULT_TRAINING_PROFILE,
    load_runtime_profile,
    validate_host_topology,
)
from qorl_training.runtime import QorlRuntime


ROOT = Path(__file__).resolve().parents[2]


class FakeWorker:
    fixture = SimpleNamespace(runtime_identity={})


class WorkerPoolTest(unittest.TestCase):
    def test_resources_are_distinct_and_parameterized(self) -> None:
        resources = load_runtime_profile(
            ROOT, DEFAULT_TRAINING_PROFILE
        ).workers

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
        profile = load_runtime_profile(ROOT, DEFAULT_TRAINING_PROFILE)
        slots = tuple(
            WorkerSlot(resources, FakeWorker())  # type: ignore[arg-type]
            for resources in profile.workers
        )
        runtime = QorlRuntime(  # type: ignore[arg-type]
            SimpleNamespace(data_identity={}),
            slots,
            "test-pool",
            "test-sha",
        )

        with runtime.claim_worker() as first:
            with runtime.claim_worker() as second:
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

            profile = load_runtime_profile(ROOT, DEFAULT_TRAINING_PROFILE)
            validate_host_topology(profile.workers, root)

            with self.assertRaisesRegex(RuntimeError, "physical cores"):
                validate_host_topology(
                    (replace(profile.workers[0], cpuset="0-1"),),
                    root,
                )


if __name__ == "__main__":
    unittest.main()
