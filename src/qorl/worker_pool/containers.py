from __future__ import annotations

import contextlib
import os
import queue
import re
import subprocess
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path, PurePosixPath

from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.worker_pool.config import validate_host_topology
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import PoolConfig, WorkerSlot

STOP_TIMEOUT_SECONDS = 60


class ContainerPool:
    """Manage container lifecycles and lend their SQL clients to callers."""

    def __init__(
        self,
        repository: Path,
        project_name: str,
        pool_config: PoolConfig,
        postgres_config: PostgresConfig,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project_name):
            raise ValueError("invalid Compose project name")
        self.repository = repository.resolve()
        self.project_name = project_name
        self.pool_config = pool_config
        self.postgres_config = postgres_config
        self.workers = tuple(
            WorkerSlot(resources, f"{project_name}-{resources.index}")
            for resources in pool_config.workers
        )
        self._available: queue.Queue[WorkerSlot] = queue.Queue()
        for slot in self.workers:
            slot.client = PostgresClient(partial(self.exec, slot))
            self._available.put(slot)

    def execute(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            cwd=self.repository,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ContainerError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
            )
        return completed

    def command(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> str:
        return self.execute(command, input_text=input_text, check=check).stdout

    def exec(
        self, slot: WorkerSlot, command: list[str], input_text: str | None
    ) -> subprocess.CompletedProcess[str]:
        return self.execute(
            ["docker", "exec", "--interactive", slot.container_id, *command],
            input_text=input_text,
            check=False,
        )

    def compose_command(
        self, slot: WorkerSlot, *arguments: str, check: bool = True
    ) -> str:
        resources = slot.resources
        config = self.postgres_config
        environment = {
            **os.environ,
            "QORL_POSTGRES_CPUSET": resources.cpuset,
            "QORL_POSTGRES_CPUSET_MEMS": resources.cpuset_mems,
            "QORL_POSTGRES_MEMORY_LIMIT": resources.memory_limit,
            "QORL_POSTGRES_MEMORY_BYTES": str(resources.memory_bytes),
            "QORL_POSTGRES_MEMORY_SWAP_LIMIT": resources.memory_limit,
            "QORL_POSTGRES_MEMORY_SWAP_BYTES": str(resources.memory_swap_bytes),
            "QORL_POSTGRES_SHM_SIZE": resources.shm_size,
            "QORL_POSTGRES_SHM_BYTES": str(resources.shm_bytes),
            "QORL_POSTGRES_PORT": str(resources.port),
            "QORL_POSTGRES_CONFIG_FILE": str(config.pg_conf_path),
            "QORL_POSTGRES_EXPECTED_FILE": str(config.expected_path),
            "QORL_POSTGRES_ASSERT_SCRIPT": str(
                self.repository / "docker/postgres/scripts/assert-config.sh"
            ),
            "QORL_POSTGRES_DUMP_SCRIPT": str(
                self.repository / "docker/postgres/scripts/dump-postgres-state.sh"
            ),
        }
        return self.execute(
            [
                "docker",
                "compose",
                "--project-name",
                slot.compose_project_name,
                "--file",
                str(self.repository / "compose.yaml"),
                *arguments,
            ],
            check=check,
            environment=environment,
        ).stdout

    def create(self) -> None:
        validate_host_topology(self.pool_config.workers)
        for slot in self.workers:
            self._create(slot)

    def _create(self, slot: WorkerSlot) -> None:
        self.compose_command(slot, "config", "--quiet")
        if self.compose_command(slot, "ps", "--all", "--quiet").strip():
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
        self.compose_command(slot, "create", "--no-build", "postgres")
        slot.created = True
        slot.container_id = self.compose_command(
            slot, "ps", "--all", "--quiet", "postgres"
        ).strip()
        if not slot.container_id:
            raise ContainerError("Docker did not create the PostgreSQL container")
        slot.volume = self.command(
            [
                "docker",
                "inspect",
                slot.container_id,
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}',
            ]
        ).strip()
        if not slot.volume:
            raise ContainerError("Docker did not create the PostgreSQL data volume")
        slot.image_id = self.command(
            ["docker", "inspect", slot.container_id, "--format", "{{.Image}}"]
        ).strip()
        environment = self.command(
            [
                "docker",
                "inspect",
                slot.container_id,
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
            ]
        )
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
        for slot in self.workers:
            if not slot.created:
                raise ContainerError("create the container before restoring IMDb")
            self._restore(slot, archive)

    def _restore(self, slot: WorkerSlot, archive: Path) -> None:
        restore_script = r"""
test -z "$(find /target -mindepth 1 -print -quit)"
mkdir -p "/target/$2"
gzip --decompress --stdout "/archive/$1" \
    | tar --extract --directory="/target/$2" --numeric-owner
test -f "/target/$2/PG_VERSION"
test ! -e "/target/$2/postmaster.pid"
"""
        self.command(
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
        for slot in self.workers:
            if not slot.created:
                raise ContainerError("create the container before starting PostgreSQL")
            self.compose_command(
                slot, "up", "--detach", "--wait", "--no-build", "postgres"
            )
            self.command(["docker", "exec", slot.container_id, "qorl-assert-config"])

    def stop(self) -> None:
        for slot in self.workers:
            self.compose_command(
                slot, "stop", "--timeout", str(STOP_TIMEOUT_SECONDS), "postgres"
            )

    def close(self) -> None:
        for slot in reversed(self.workers):
            if slot.created:
                self.compose_command(slot, "down", "--volumes", check=False)
                slot.created = False
                slot.container_id = ""

    @contextlib.contextmanager
    def claim_worker(self) -> Iterator[WorkerSlot]:
        slot = self._available.get()
        try:
            yield slot
        finally:
            self._available.put(slot)

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.pool_config.profile_id,
            "path": str(self.pool_config.path),
            "config_sha256": self.pool_config.sha256,
            "worker_count": len(self.workers),
            "workers": [slot.resources.manifest() for slot in self.workers],
            "postgres_config": self.postgres_config.manifest().model_dump(),
        }

    @property
    def runtime_identity(self) -> dict[str, str]:
        return self.postgres_config.runtime_identity().model_dump(exclude_none=True)


def start_pool(
    repository: Path,
    project_name: str,
    archive: Path,
    *,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> ContainerPool:
    if not archive.is_file():
        raise ContainerError(f"database archive is missing: {archive}")
    pool = ContainerPool(
        repository,
        project_name,
        pool_config,
        postgres_config,
    )
    try:
        pool.create()
        pool.restore(archive)
        pool.start()
    except BaseException:
        pool.close()
        raise
    return pool
