from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qorl.postgres.config import PostgresConfig
from qorl.rl import runtime
from qorl.rl.runtime import QorlRuntime
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import (
    load_pool_config,
    validate_host_topology,
)

ROOT = Path(__file__).resolve().parents[3]


class WorkerPoolTest(unittest.TestCase):
    def test_start_loads_indexes_after_restoring_and_starting_postgres(self) -> None:
        with (
            patch.object(runtime, "_runtime", None),
            patch.object(runtime, "QorlRuntime") as factory,
        ):
            started = runtime.start(
                ROOT,
                TaskSet.load(ROOT, "job"),
                PostgresConfig.load(Path("docker/postgres/configs/000-pgconf-default")),
                load_pool_config(Path("docker/worker_pool/configs/002-poolconf-4x8")),
            )
            self.assertIs(started, factory.return_value)
            self.assertEqual(
                [call[0] for call in factory.return_value.method_calls],
                ["create", "restore", "start", "load_indexes"],
            )

    def test_failed_start_closes_owned_pool(self) -> None:
        with (
            patch.object(runtime, "_runtime", None),
            patch.object(runtime, "QorlRuntime") as factory,
        ):
            factory.return_value.restore.side_effect = RuntimeError("restore failed")
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                runtime.start(
                    ROOT,
                    TaskSet.load(ROOT, "job"),
                    PostgresConfig.load(
                        Path("docker/postgres/configs/000-pgconf-default")
                    ),
                    load_pool_config(
                        Path("docker/worker_pool/configs/002-poolconf-4x8")
                    ),
                )
            factory.return_value.close.assert_called_once()
            with self.assertRaises(RuntimeError):
                runtime.current()

    def test_resources_are_distinct_and_parameterized(self) -> None:
        resources = load_pool_config(
            Path("docker/worker_pool/configs/002-poolconf-4x8")
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
        profile = load_pool_config(Path("docker/worker_pool/configs/002-poolconf-4x8"))
        runtime = QorlRuntime(
            TaskSet.load(ROOT, "ceb"),
            profile,
            "test-pool",
            PostgresConfig.load(Path("docker/postgres/configs/000-pgconf-default")),
        )
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.compose_project_name, "test-pool")

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
                Path("docker/worker_pool/configs/002-poolconf-4x8")
            )
            validate_host_topology(profile.workers, root)

            with self.assertRaisesRegex(RuntimeError, "physical cores"):
                validate_host_topology(
                    (replace(profile.workers[0], cpuset="0-1"),),
                    root,
                )


if __name__ == "__main__":
    unittest.main()


def test_drain_waits_for_all_episode_threads() -> None:
    import asyncio
    from threading import Event

    pool = QorlRuntime(
        TaskSet.load(ROOT, "job"),
        load_pool_config(Path("docker/worker_pool/configs/002-poolconf-4x8")),
        "drain-test",
        PostgresConfig.load(Path("docker/postgres/configs/000-pgconf-default")),
    )

    async def check() -> None:
        cancelled, release, finished = Event(), Event(), Event()

        def execute() -> None:
            assert cancelled.wait(2)
            assert release.wait(2)
            finished.set()

        work = asyncio.create_task(asyncio.to_thread(execute))
        pool.work[work] = cancelled
        with patch.object(runtime, "_runtime", pool):
            cleaning = asyncio.create_task(runtime.drain())
            assert await asyncio.to_thread(cancelled.wait, 2)
            cleaning.cancel()
            await asyncio.sleep(0)
            assert not cleaning.done() and not finished.is_set()
            release.set()
            await cleaning
            await work
            assert finished.is_set()

    asyncio.run(check())
