from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from qorl.db.config import PostgresConfig
from qorl.db.exceptions import WorkerError
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import RuntimeProfile, WorkerResources, validate_host_topology


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
    """Own one restored PostgreSQL container and its Docker lifecycle."""

    def __init__(
        self,
        fixture: DatabaseFixture,
        project_name: str,
        runtime_profile: RuntimeProfile,
        resources: WorkerResources,
        postgres_config: PostgresConfig | None = None,
    ) -> None:
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

    def start(self) -> None:
        validate_host_topology((self.resources,))
        fixture_id = self.fixture.manifest["fixture_id"]
        print(f"Verifying {fixture_id} database archive...")
        self.fixture.verify_archive()

        image = self.fixture.manifest["image"]
        actual_image_id = self.command(
            ["docker", "image", "inspect", image["reference"], "--format", "{{.Id}}"]
        ).strip()
        if actual_image_id != image["id"]:
            raise WorkerError(
                f"database image mismatch: expected={image['id']} "
                f"actual={actual_image_id}"
            )

        self.compose_command("config", "--quiet")
        if self.compose_command("ps", "--all", "--quiet").strip():
            raise WorkerError(f"Docker project already exists: {self.project_name}")

        print("Restoring IMDb into a fresh Docker volume...")
        self.compose_command("create", "--no-build", "postgres")
        self.created = True
        self.container = self.compose_command(
            "ps", "--all", "--quiet", "postgres"
        ).strip()
        if not self.container:
            raise WorkerError("Docker did not create the PostgreSQL container")

        volume = self.command(
            [
                "docker",
                "inspect",
                self.container,
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}',
            ]
        ).strip()
        if not volume:
            raise WorkerError("Docker did not create the PostgreSQL data volume")

        archive = self.fixture.archive_path
        relative = PurePosixPath(
            self.fixture.manifest["postgresql"]["pgdata_volume_relative_path"]
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkerError("database manifest contains an invalid PGDATA path")
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
                f"{volume}:/target",
                "--volume",
                f"{archive.parent}:/archive:ro",
                "--entrypoint",
                "bash",
                image["id"],
                "-Eeuo",
                "pipefail",
                "-c",
                restore_script,
                "qorl-restore",
                archive.name,
                str(relative),
            ]
        )

        print("Starting PostgreSQL worker...")
        self.compose_command("up", "--detach", "--wait", "--no-build", "postgres")
        self.container = self.compose_command("ps", "--quiet", "postgres").strip()
        self.command(["docker", "exec", self.container, "qorl-assert-config"])

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
