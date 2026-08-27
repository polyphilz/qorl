from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from qprl.fixture import JobFixture


class WorkerError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkerError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed.stdout


class PostgresWorker:
    def __init__(self, fixture: JobFixture, project_name: str) -> None:
        self.fixture = fixture
        self.project_name = project_name
        self.container = ""
        self.created = False
        self.compose = [
            "docker",
            "compose",
            "--project-name",
            project_name,
            "--file",
            str(fixture.repository / "compose.yaml"),
        ]

    def __enter__(self) -> PostgresWorker:
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def command(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> str:
        return run(
            command,
            cwd=self.fixture.repository,
            input_text=input_text,
            check=check,
        )

    def compose_command(self, *arguments: str, check: bool = True) -> str:
        return self.command([*self.compose, *arguments], check=check)

    def start(self) -> None:
        print("Verifying job-v1 snapshot...")
        self.fixture.verify_archive()

        image = self.fixture.snapshot["image"]
        actual_image_id = self.command(
            ["docker", "image", "inspect", image["reference"], "--format", "{{.Id}}"]
        ).strip()
        if actual_image_id != image["id"]:
            raise WorkerError(
                f"job-v1 image mismatch: expected={image['id']} actual={actual_image_id}"
            )

        self.compose_command("config", "--quiet")
        if self.compose_command("ps", "--all", "--quiet").strip():
            raise WorkerError(f"Docker project already exists: {self.project_name}")

        print("Restoring job-v1 into a fresh Docker volume...")
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
                "{{range .Mounts}}{{if eq .Destination \"/var/lib/postgresql/data\"}}{{.Name}}{{end}}{{end}}",
            ]
        ).strip()
        if not volume:
            raise WorkerError("Docker did not create the PostgreSQL data volume")

        archive = self.fixture.archive_path
        restore_script = r"""
test -z "$(find /target -mindepth 1 -print -quit)"
gzip --decompress --stdout "/snapshot/$1" \
    | tar --extract --directory=/target --numeric-owner
test -f /target/PG_VERSION
test ! -e /target/postmaster.pid
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
                f"{archive.parent}:/snapshot:ro",
                "--entrypoint",
                "bash",
                image["id"],
                "-Eeuo",
                "pipefail",
                "-c",
                restore_script,
                "qprl-restore",
                archive.name,
            ]
        )

        print("Starting PostgreSQL worker...")
        self.compose_command("up", "--detach", "--wait", "--no-build", "postgres")
        self.container = self.compose_command("ps", "--quiet", "postgres").strip()
        self.command(
            ["docker", "exec", self.container, "qprl-assert-benchmark-config"]
        )
        actual_system_identifier = self.admin_sql(
            "SELECT system_identifier FROM pg_control_system();"
        ).strip()
        expected_system_identifier = self.fixture.snapshot["postgresql"][
            "system_identifier"
        ]
        if actual_system_identifier != expected_system_identifier:
            raise WorkerError(
                "job-v1 PostgreSQL system identifier does not match the inventory"
            )

    def close(self) -> None:
        if not self.created:
            return
        self.compose_command("down", "--volumes", check=False)
        self.created = False
        self.container = ""

    def admin_sql(self, sql: str) -> str:
        shell = r"""
exec psql \
    --username="$POSTGRES_USER" \
    --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
    --no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align
"""
        return self.command(
            [
                "docker",
                "exec",
                "--interactive",
                self.container,
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                shell,
            ],
            input_text=sql,
        )

    def explain_analyze(self, sql: str, timeout_ms: int) -> dict[str, Any]:
        shell = rf"""
exec env \
    PGPASSWORD="$QPRL_RUNNER_PASSWORD" \
    PGAPPNAME=qprl-calibration \
    PGOPTIONS="-c statement_timeout={timeout_ms}" \
    psql \
        --host=127.0.0.1 \
        --username=qprl_runner \
        --dbname="${{POSTGRES_DB:-$POSTGRES_USER}}" \
        --no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align
"""
        output = self.command(
            [
                "docker",
                "exec",
                "--interactive",
                self.container,
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                shell,
            ],
            input_text=(
                "EXPLAIN (ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON)\n"
                + sql.strip()
                + "\n"
            ),
        )
        try:
            parsed = json.loads(output)
            return parsed[0]
        except (json.JSONDecodeError, IndexError, TypeError) as error:
            raise WorkerError("PostgreSQL returned invalid EXPLAIN JSON") from error

    def capture_environment(self, output_dir: Path, phase: str) -> None:
        self.command(
            [
                sys.executable,
                str(
                    self.fixture.repository
                    / "scripts/capture-benchmark-environment.py"
                ),
                "--container",
                self.container,
                "--output-dir",
                str(output_dir),
                "--phase",
                phase,
            ]
        )
