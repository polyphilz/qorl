from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from qorl.db.config import PostgresConfig
from qorl.db.exceptions import WorkerError
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import RuntimeProfile, WorkerResources, validate_host_topology

STOP_TIMEOUT_SECONDS = 60


def execute(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkerError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


class PostgresContainer:
    """Create, populate, and run one PostgreSQL container."""

    def __init__(
        self,
        fixture: DatabaseFixture,
        project_name: str,
        runtime_profile: RuntimeProfile,
        resources: WorkerResources,
        postgres_config: PostgresConfig | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project_name):
            raise ValueError("invalid Compose project name")
        if resources not in runtime_profile.workers:
            raise ValueError("worker resources do not belong to runtime profile")
        self.fixture = fixture
        self.project_name = project_name
        self.runtime_profile = runtime_profile
        self.resources = resources
        self.postgres_config = postgres_config or PostgresConfig.load(
            fixture.repository
        )
        self.container = ""
        self.created = False
        self.volume = ""
        self.image_id = ""
        self.pgdata_relative_path = ""
        self.environment = {
            **os.environ,
            **resources.compose_environment,
            **self.postgres_config.compose_environment,
        }
        self.compose = [
            "docker",
            "compose",
            "--project-name",
            project_name,
            "--file",
            str(fixture.repository / "compose.yaml"),
        ]

    def command(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> str:
        return self.execute(
            command,
            input_text=input_text,
            check=check,
        ).stdout

    def execute(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return execute(
            command,
            cwd=self.fixture.repository,
            input_text=input_text,
            check=check,
            environment=self.environment,
        )

    def compose_command(self, *arguments: str, check: bool = True) -> str:
        return self.command([*self.compose, *arguments], check=check)

    def create(self) -> None:
        validate_host_topology((self.resources,))
        self.compose_command("config", "--quiet")
        if self.compose_command("ps", "--all", "--quiet").strip():
            raise WorkerError(f"Docker project already exists: {self.project_name}")
        volume = f"{self.project_name}_qorl-postgres-data"
        if (
            self.execute(
                ["docker", "volume", "inspect", volume], check=False
            ).returncode
            == 0
        ):
            raise WorkerError(f"Docker volume already exists: {volume}")

        self.compose_command("create", "--no-build", "postgres")
        self.created = True
        self.container = self.compose_command(
            "ps", "--all", "--quiet", "postgres"
        ).strip()
        if not self.container:
            raise WorkerError("Docker did not create the PostgreSQL container")
        self.volume = self.command(
            [
                "docker",
                "inspect",
                self.container,
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}',
            ]
        ).strip()
        if not self.volume:
            raise WorkerError("Docker did not create the PostgreSQL data volume")
        self.image_id = self.command(
            [
                "docker",
                "inspect",
                self.container,
                "--format",
                "{{.Image}}",
            ]
        ).strip()
        environment = self.command(
            [
                "docker",
                "inspect",
                self.container,
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
            raise WorkerError("invalid PGDATA path in PostgreSQL image")
        self.pgdata_relative_path = str(path.relative_to("/var/lib/postgresql"))

    def restore_archive(self) -> None:
        if not self.created:
            raise WorkerError("create the container before restoring IMDb")
        archive = self.fixture.archive_path
        if not archive.is_file():
            raise WorkerError(f"database archive is missing: {archive}")
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
                f"{self.volume}:/target",
                "--volume",
                f"{archive.parent}:/archive:ro",
                "--entrypoint",
                "bash",
                self.image_id,
                "-Eeuo",
                "pipefail",
                "-c",
                restore_script,
                "qorl-restore",
                archive.name,
                self.pgdata_relative_path,
            ]
        )

    def start(self) -> None:
        if not self.created:
            raise WorkerError("create the container before starting PostgreSQL")
        self.compose_command("up", "--detach", "--wait", "--no-build", "postgres")
        self.command(["docker", "exec", self.container, "qorl-assert-config"])

    def stop(self) -> None:
        self.compose_command("stop", "--timeout", str(STOP_TIMEOUT_SECONDS), "postgres")

    def close(self) -> None:
        if not self.created:
            return
        self.compose_command("down", "--volumes", check=False)
        self.created = False
        self.container = ""

    def capture_environment(self, output_dir: Path, phase: str) -> None:
        self.command(
            [
                sys.executable,
                "-m",
                "qorl.db.capture",
                "--container",
                self.container,
                "--output-dir",
                str(output_dir),
                "--phase",
                phase,
                "--runtime-profile",
                str(self.fixture.repository / self.runtime_profile.path),
                "--postgres-config",
                str(self.postgres_config.path),
            ]
        )
