from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qorl.postgres.config import PostgresConfig
from qorl.taskset.taskset import TaskSet
from qorl.training import runtime
from qorl.training.runtime import QorlRuntime
from qorl.worker_pool.config import (
    load_pool_config,
    validate_host_topology,
)

ROOT = Path(__file__).resolve().parents[3]


class WorkerPoolTest(unittest.TestCase):
    def test_start_requires_both_configuration_paths(self) -> None:
        for missing in (runtime.POSTGRES_CONFIG_ENV, runtime.POOL_CONFIG_ENV):
            for value in (None, "", " "):
                with self.subTest(missing=missing, value=value):
                    environment = {
                        runtime.POSTGRES_CONFIG_ENV: "docker/postgres/configs/000-pgconf-default",
                        runtime.POOL_CONFIG_ENV: "docker/worker_pool/configs/002-poolconf-4x8",
                    }
                    if value is None:
                        del environment[missing]
                    else:
                        environment[missing] = value
                    with patch.object(runtime, "QorlRuntime") as pool:
                        with self.assertRaisesRegex(RuntimeError, missing):
                            runtime.start(ROOT, environment)
                        pool.assert_not_called()

    def test_resources_are_distinct_and_parameterized(self) -> None:
        resources = load_pool_config(
            ROOT, Path("docker/worker_pool/configs/002-poolconf-4x8")
        ).workers

        self.assertEqual([item.index for item in resources], [0, 1, 2, 3])
        self.assertEqual(
            resources[0].cpuset,
            "0-3,16-19",
        )
        self.assertEqual(
            resources[3].port,
            56003,
        )
        self.assertEqual(resources[0].memory_bytes, 8 * 1024**3)

    def test_claim_returns_workers_to_the_pool(self) -> None:
        profile = load_pool_config(
            ROOT, Path("docker/worker_pool/configs/002-poolconf-4x8")
        )
        runtime = QorlRuntime(
            ROOT,
            TaskSet.load(ROOT, "ceb"),
            profile,
            "test-pool",
            PostgresConfig.load(
                ROOT, Path("docker/postgres/configs/000-pgconf-default")
            ),
        )
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.repository, ROOT.resolve())

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

            profile = load_pool_config(
                ROOT, Path("docker/worker_pool/configs/002-poolconf-4x8")
            )
            validate_host_topology(profile.workers, root)

            with self.assertRaisesRegex(RuntimeError, "physical cores"):
                validate_host_topology(
                    (replace(profile.workers[0], cpuset="0-1"),),
                    root,
                )


if __name__ == "__main__":
    unittest.main()
