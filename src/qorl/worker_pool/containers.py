from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path, PurePosixPath

from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes
from qorl.worker_pool.config import validate_host_topology
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import (
    ComposeEnvironment,
    PoolConfig,
    PoolManifest,
    WorkerSlot,
)

STOP_TIMEOUT_SECONDS = 60
logger = logging.getLogger(__name__)


class ContainerPool:
    """Manage container lifecycles and lend their SQL clients to callers."""

    def __init__(
        self,
        compose_project_name: str,
        pool_config: PoolConfig,
        postgres_config: PostgresConfig,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", compose_project_name):
            raise ValueError("invalid Compose project name")
        self.compose_project_name = compose_project_name
        self.pool_config = pool_config
        self.postgres_config = postgres_config
        self.settings = postgres_config.agent_settings
        self.indexes = PostgresIndexes(by_table={})
        self.workers = tuple(
            WorkerSlot(resources, f"{compose_project_name}-{resources.index}")
            for resources in pool_config.workers
        )
        self._available: queue.Queue[WorkerSlot] = queue.Queue()
        for slot in self.workers:
            slot.client = PostgresClient(
                partial(self.exec, slot), self.settings, self.indexes
            )
            self._available.put(slot)

    def execute(
        self,
        command: list[str],
        *,
        stdin: str | None = None,
        check: bool = True,
        environment: ComposeEnvironment | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command, supplying stdin text when given.

        With no stdin or environment overrides, inherit the process's input and
        environment. Compose overrides are merged with the OS environment.
        """
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            env=None if environment is None else {**os.environ, **environment.to_env()},
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ContainerError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
            )
        return completed

    def exec(
        self,
        slot: WorkerSlot,
        command: list[str],
        query: str,
    ) -> subprocess.CompletedProcess[str]:
        return self.execute(
            ["docker", "exec", "--interactive", slot.container_id, *command],
            stdin=query,
            check=False,
        )

    def compose_command(
        self,
        slot: WorkerSlot,
        arguments: list[str],
    ) -> str:
        """Run Compose with typed per-worker overrides and raise on failure."""
        resources = slot.resources
        config = self.postgres_config
        environment = ComposeEnvironment(
            cpuset=resources.cpuset,
            cpuset_mems=resources.cpuset_mems,
            memory_limit=resources.memory_limit,
            memory_bytes=resources.memory_bytes,
            memory_swap_limit=resources.memory_limit,
            memory_swap_bytes=resources.memory_swap_bytes,
            shm_size=resources.shm_size,
            shm_bytes=resources.shm_bytes,
            port=resources.port,
            config_file=config.pg_conf_path,
            expected_file=config.expected_path,
            assert_script=REPOSITORY_ROOT / "docker/postgres/scripts/assert-config.sh",
            dump_script=REPOSITORY_ROOT
            / "docker/postgres/scripts/dump-postgres-state.sh",
        )
        return self.execute(
            [
                "docker",
                "compose",
                "--project-name",
                slot.compose_project_name,
                "--file",
                str(REPOSITORY_ROOT / "compose.yaml"),
                *arguments,
            ],
            environment=environment,
        ).stdout

    def _parallel(self, operation: Callable[[WorkerSlot], None]) -> None:
        """Finish active operations before propagating failure or allowing cleanup."""
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(operation, slot) for slot in self.workers]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    def create(self) -> None:
        validate_host_topology(self.pool_config.workers)
        self._parallel(self._create)

    def _create(self, slot: WorkerSlot) -> None:
        self.compose_command(slot, ["config", "--quiet"])
        if self.compose_command(slot, ["ps", "--all", "--quiet"]).strip():
            raise ContainerError(
                f"Docker project already exists: {slot.compose_project_name}"
            )
        volume = f"{slot.compose_project_name}_qorl-postgres-data"
        if (
            self.execute(
                ["docker", "volume", "inspect", volume], check=False
            ).returncode
            == 0
        ):
            raise ContainerError(f"Docker volume already exists: {volume}")
        # A failed Compose create can still leave resources that need cleanup.
        slot.created = True
        self.compose_command(slot, ["create", "--no-build", "postgres"])
        slot.container_id = self.compose_command(
            slot, ["ps", "--all", "--quiet", "postgres"]
        ).strip()
        if not slot.container_id:
            raise ContainerError("Docker did not create the PostgreSQL container")
        slot.volume = self.execute(
            [
                "docker",
                "inspect",
                slot.container_id,
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}',
            ]
        ).stdout.strip()
        if not slot.volume:
            raise ContainerError("Docker did not create the PostgreSQL data volume")
        slot.image_id = self.execute(
            ["docker", "inspect", slot.container_id, "--format", "{{.Image}}"]
        ).stdout.strip()
        environment = self.execute(
            [
                "docker",
                "inspect",
                slot.container_id,
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
            ]
        ).stdout
        pgdata = next(
            (
                line.removeprefix("PGDATA=")
                for line in environment.splitlines()
                if line.startswith("PGDATA=")
            ),
            "",
        )
        path = PurePosixPath(pgdata)
        if ".." in path.parts:
            raise ContainerError("invalid PGDATA path in PostgreSQL image")
        slot.pgdata_relative_path = str(path.relative_to("/var/lib/postgresql"))

    def restore(self, archive: Path) -> None:
        archive = archive.resolve()
        if not archive.is_file():
            raise ContainerError(f"database archive is missing: {archive}")
        if not all(slot.created for slot in self.workers):
            raise ContainerError("create the container before restoring IMDb")
        self._parallel(partial(self._restore, archive=archive))

    def _restore(self, slot: WorkerSlot, archive: Path) -> None:
        restore_script = r"""
test -z "$(find /target -mindepth 1 -print -quit)"
mkdir -p "/target/$2"
gzip --decompress --stdout "/archive/$1" \
    | tar --extract --directory="/target/$2" --numeric-owner
test -f "/target/$2/PG_VERSION"
test ! -e "/target/$2/postmaster.pid"
"""
        self.execute(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--volume",
                f"{slot.volume}:/target",
                "--volume",
                f"{archive.parent}:/archive:ro",
                "--entrypoint",
                "bash",
                slot.image_id,
                "-Eeuo",
                "pipefail",
                "-c",
                restore_script,
                "qorl-restore",
                archive.name,
                slot.pgdata_relative_path,
            ]
        )

    def start(self) -> None:
        if not all(slot.created for slot in self.workers):
            raise ContainerError("create the container before starting PostgreSQL")
        self._parallel(self._start)

    def _start(self, slot: WorkerSlot) -> None:
        self.compose_command(
            slot, ["up", "--detach", "--wait", "--no-build", "postgres"]
        )
        self.execute(["docker", "exec", slot.container_id, "qorl-assert-config"])

    def load_indexes(self) -> None:
        """Populate the shared catalog from one worker after restoring or loading IMDb."""
        self.indexes.by_table = self.workers[0].client.read_indexes().by_table

    def stop(self) -> None:
        for slot in self.workers:
            self.compose_command(
                slot, ["stop", "--timeout", str(STOP_TIMEOUT_SECONDS), "postgres"]
            )

    def close(self) -> None:
        """Attempt every owned project's removal; log failures and retain them for retry."""
        for slot in reversed(self.workers):
            if slot.created:
                try:
                    self.compose_command(slot, ["down", "--volumes"])
                except (ContainerError, OSError) as error:
                    logger.error(
                        "Cleanup failed for %s: %s", slot.compose_project_name, error
                    )
                    continue
                slot.created = False
                slot.container_id = ""

    @contextlib.contextmanager
    def claim_worker(self) -> Iterator[WorkerSlot]:
        slot = self._available.get()
        try:
            yield slot
        finally:
            self._available.put(slot)

    def manifest(self) -> PoolManifest:
        return PoolManifest(
            id=self.pool_config.profile_id,
            path=str(self.pool_config.path),
            config_sha256=self.pool_config.sha256,
            worker_count=len(self.workers),
            workers=[slot.resources.manifest() for slot in self.workers],
            postgres_config=self.postgres_config.manifest(),
        )


def start_pool(
    compose_project_name: str,
    archive: Path,
    *,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> ContainerPool:
    if not archive.is_file():
        raise ContainerError(f"database archive is missing: {archive}")
    pool = ContainerPool(
        compose_project_name,
        pool_config,
        postgres_config,
    )
    try:
        pool.create()
        pool.restore(archive)
        pool.start()
        pool.load_indexes()
    except BaseException:
        pool.close()
        raise
    return pool
