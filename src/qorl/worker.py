from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qorl.fixture import DatabaseFixture


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExplainResult:
    document: dict[str, Any]
    hint_diagnostics: str


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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkerError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def run(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> str:
    return execute(
        command,
        cwd=cwd,
        input_text=input_text,
        check=check,
        environment=environment,
    ).stdout


class PostgresWorker:
    def __init__(
        self,
        fixture: DatabaseFixture,
        project_name: str,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.fixture = fixture
        self.project_name = project_name
        self.container = ""
        self.created = False
        self.explain_calls = 0
        self.explain_analyze_calls = 0
        self.environment = (
            {**os.environ, **environment} if environment is not None else None
        )
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
            environment=self.environment,
        )

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
        fixture_id = self.fixture.snapshot["fixture_id"]
        print(f"Verifying {fixture_id} database snapshot...")
        self.fixture.verify_archive()

        image = self.fixture.snapshot["image"]
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
                "{{range .Mounts}}{{if eq .Destination \"/var/lib/postgresql\"}}{{.Name}}{{end}}{{end}}",
            ]
        ).strip()
        if not volume:
            raise WorkerError("Docker did not create the PostgreSQL data volume")

        archive = self.fixture.archive_path
        relative = PurePosixPath(
            self.fixture.snapshot["postgresql"]["pgdata_volume_relative_path"]
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkerError("database snapshot contains an invalid PGDATA path")
        restore_script = r"""
test -z "$(find /target -mindepth 1 -print -quit)"
mkdir -p "/target/$2"
gzip --decompress --stdout "/snapshot/$1" \
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
                f"{archive.parent}:/snapshot:ro",
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
        self.command(
            ["docker", "exec", self.container, "qorl-assert-benchmark-config"]
        )
        actual_system_identifier = self.admin_sql(
            "SELECT system_identifier FROM pg_control_system();"
        ).strip()
        expected_system_identifier = self.fixture.snapshot["postgresql"][
            "system_identifier"
        ]
        if actual_system_identifier != expected_system_identifier:
            raise WorkerError(
                "PostgreSQL system identifier does not match the database snapshot"
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

    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        self.explain_calls += 1
        self.explain_analyze_calls += int(analyze)
        explain_options = (
            "ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON"
            if analyze
            else "FORMAT JSON"
        )
        debug_options = (
            " -c pg_hint_plan.debug_print=detailed"
            " -c pg_hint_plan.message_level=notice"
            if hint
            else ""
        )
        shell = rf"""
exec env \
    PGPASSWORD="$QORL_RUNNER_PASSWORD" \
    PGAPPNAME=qorl-worker \
    PGOPTIONS="-c statement_timeout={timeout_ms}{debug_options}" \
    psql \
        --host=127.0.0.1 \
        --username=qorl_runner \
        --dbname="${{POSTGRES_DB:-$POSTGRES_USER}}" \
        --no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align
"""
        completed = self.execute(
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
                f"EXPLAIN ({explain_options})\n"
                + (hint + "\n" if hint else "")
                + sql.strip()
                + "\n"
            ),
        )
        try:
            parsed = json.loads(completed.stdout)
            return ExplainResult(parsed[0], completed.stderr)
        except (json.JSONDecodeError, IndexError, TypeError) as error:
            raise WorkerError("PostgreSQL returned invalid EXPLAIN JSON") from error

    def explain_analyze(self, sql: str, timeout_ms: int) -> dict[str, Any]:
        return self.explain(sql, timeout_ms, analyze=True).document

    def task_indexes(self, task: dict[str, Any]) -> dict[str, set[str]]:
        tables = sorted({relation["table"] for relation in task["relations"]})
        literals = ", ".join("'" + table.replace("'", "''") + "'" for table in tables)
        output = self.admin_sql(
            "SELECT json_object_agg(tablename, indexes ORDER BY tablename) "
            "FROM ("
            "SELECT tablename, json_agg(indexname ORDER BY indexname) AS indexes "
            "FROM pg_indexes "
            f"WHERE schemaname = 'public' AND tablename IN ({literals}) "
            "GROUP BY tablename"
            ") AS listed;"
        ).strip()
        by_table = json.loads(output)
        return {
            relation["alias"]: set(by_table.get(relation["table"], []))
            for relation in task["relations"]
        }

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
