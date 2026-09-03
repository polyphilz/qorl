from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import (
    DEFAULT_EVALUATION_PROFILE,
    RuntimeProfile,
    WorkerResources,
    load_runtime_profile,
    validate_host_topology,
)


class WorkerError(RuntimeError):
    pass


class QueryTimeout(WorkerError):  # noqa: N818
    def __init__(self, timeout_ms: int) -> None:
        super().__init__(f"query exceeded statement_timeout={timeout_ms} ms")
        self.timeout_ms = timeout_ms


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
        runtime_profile: RuntimeProfile | None = None,
        resources: WorkerResources | None = None,
    ) -> None:
        self.fixture = fixture
        self.project_name = project_name
        self.container = ""
        self.created = False
        self.explain_calls = 0
        self.explain_analyze_calls = 0
        self._settings_cache: dict[str, str] = {}
        profile = runtime_profile or load_runtime_profile(
            fixture.repository, DEFAULT_EVALUATION_PROFILE
        )
        if resources is None:
            if len(profile.workers) != 1:
                raise ValueError(
                    "a multi-worker runtime profile requires an explicit slot"
                )
            resources = profile.workers[0]
        elif resources not in profile.workers:
            raise ValueError("worker resources do not belong to runtime profile")
        self.runtime_profile = profile
        self.resources = resources
        self.environment = {**os.environ, **resources.compose_environment}
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
        validate_host_topology((self.resources,))
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
                '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}',
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
        self.command(["docker", "exec", self.container, "qorl-assert-benchmark-config"])
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

    def runner_sql(self, sql: str) -> str:
        shell = r"""
exec env \
    PGPASSWORD="$QORL_RUNNER_PASSWORD" \
    PGAPPNAME=qorl-worker \
    psql \
        --host=127.0.0.1 \
        --username=qorl_runner \
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

    def settings(self, names: set[str]) -> dict[str, str]:
        missing = names - self._settings_cache.keys()
        if not missing:
            return {name: self._settings_cache[name] for name in sorted(names)}
        ordered = sorted(missing)
        literals = ", ".join("'" + name.replace("'", "''") + "'" for name in ordered)
        output = self.runner_sql(
            "SELECT json_object_agg(name, setting ORDER BY name) "
            "FROM pg_settings "
            f"WHERE name IN ({literals});"
        ).strip()
        values = json.loads(output)
        if not isinstance(values, dict) or set(values) != missing:
            raise WorkerError("PostgreSQL planner-setting response is incomplete")
        self._settings_cache.update({name: str(values[name]) for name in ordered})
        return {name: self._settings_cache[name] for name in sorted(names)}

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
            "ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON" if analyze else "FORMAT JSON"
        )
        debug_options = (
            " -c pg_hint_plan.debug_print=detailed -c pg_hint_plan.message_level=notice"
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
        try:
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
        except WorkerError as error:
            if "canceling statement due to statement timeout" in str(error):
                raise QueryTimeout(timeout_ms) from error
            raise
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
            ]
        )

    def runtime_manifest(self) -> dict[str, object]:
        return {
            "profile": {
                "id": self.runtime_profile.profile_id,
                "path": str(self.runtime_profile.path),
                "sha256": self.runtime_profile.sha256,
            },
            "worker": self.resources.manifest(),
        }
