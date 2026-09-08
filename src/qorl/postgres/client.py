from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable

from pydantic import ValidationError

from qorl.postgres.exceptions import PostgresError, QueryTimeoutError
from qorl.postgres.schemas import (
    ExplainResult,
    PostgresIndexes,
    PostgresSettings,
    WorkerAllocation,
)


class PostgresClient:
    """Run SQL and retrieve query plans, timings, and database metadata."""

    def __init__(
        self,
        run_command: Callable[[list[str], str], subprocess.CompletedProcess[str]],
        settings: PostgresSettings,
        indexes: PostgresIndexes,
    ) -> None:
        self._run_command = run_command
        self.settings = settings
        self.indexes = indexes
        self.allocation: WorkerAllocation | None = None
        self.explain_calls = 0
        self.explain_analyze_calls = 0

    def execute(
        self,
        command: list[str],
        *,
        query: str,
    ) -> subprocess.CompletedProcess[str]:
        completed = self._run_command(command, query)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PostgresError(
                f"command failed ({completed.returncode}): "
                f"{' '.join(completed.args)}\n{detail}"
            )
        return completed

    def admin_sql(self, sql: str) -> str:
        """Run setup and metadata SQL as the privileged PostgreSQL administrator."""
        shell = r"""
exec psql \
    --username="$POSTGRES_USER" \
    --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
    --no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align
"""
        return self.execute(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                shell,
            ],
            query=sql,
        ).stdout

    def _run_runner_sql(
        self,
        sql: str,
        *,
        output_flags: list[str],
        connection_label: str = "qorl-worker",
        postgres_options: str = "",
    ) -> subprocess.CompletedProcess[str]:
        """Execute SQL as qorl_runner and retain both output and diagnostics."""
        environment = [f"PGAPPNAME={connection_label}"]
        if postgres_options:
            environment.append(f"PGOPTIONS={postgres_options}")
        shell = rf"""
exec env \
    PGPASSWORD="$QORL_RUNNER_PASSWORD" \
    {shlex.join(environment)} \
    psql \
        --host=127.0.0.1 \
        --username=qorl_runner \
        --dbname="${{POSTGRES_DB:-$POSTGRES_USER}}" \
        --no-psqlrc --set=ON_ERROR_STOP=1 --quiet {shlex.join(output_flags)}
"""
        return self.execute(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                shell,
            ],
            query=sql,
        )

    def runner_sql(
        self,
        sql: str,
        *,
        connection_label: str = "qorl-worker",
    ) -> str:
        """Return CSV from qorl_runner, with SELECT permissions and read-only defaults."""
        return self._run_runner_sql(
            sql,
            output_flags=["--csv"],
            connection_label=connection_label,
        ).stdout

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
        try:
            completed = self._run_runner_sql(
                (
                    f"EXPLAIN ({explain_options})\n"
                    + (hint + "\n" if hint else "")
                    + sql.strip()
                    + "\n"
                ),
                output_flags=["--tuples-only", "--no-align"],
                postgres_options=f"-c statement_timeout={timeout_ms}{debug_options}",
            )
        except PostgresError as error:
            if "canceling statement due to statement timeout" in str(error):
                raise QueryTimeoutError(timeout_ms) from error
            raise
        try:
            parsed = json.loads(completed.stdout)
            return ExplainResult(parsed[0], completed.stderr)
        except (json.JSONDecodeError, IndexError, TypeError) as error:
            raise PostgresError("PostgreSQL returned invalid EXPLAIN JSON") from error

    def read_indexes(self) -> PostgresIndexes:
        """Read all public-schema index names for the pool's shared catalog."""
        output = self.admin_sql(
            "SELECT json_build_object('by_table', "
            "COALESCE(json_object_agg(tablename, indexes ORDER BY tablename), '{}'::json)) "
            "FROM ("
            "SELECT tablename, json_agg(indexname ORDER BY indexname) AS indexes "
            "FROM pg_indexes "
            "WHERE schemaname = 'public' "
            "GROUP BY tablename"
            ") AS listed;"
        ).strip()
        try:
            return PostgresIndexes.model_validate_json(output)
        except ValidationError as error:
            raise PostgresError("PostgreSQL returned invalid index metadata") from error
