from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from qorl.db.container import PostgresContainer
from qorl.db.exceptions import QueryTimeout, WorkerError


@dataclass(frozen=True)
class ExplainResult:
    document: dict[str, Any]
    hint_diagnostics: str


class PostgresWorker:
    """Execute inspected and measured SQL against one running container."""

    def __init__(self, container: PostgresContainer) -> None:
        self._container = container
        self.fixture = container.fixture
        self.explain_calls = 0
        self.explain_analyze_calls = 0
        self._settings_cache: dict[str, str] = {}

    @property
    def container(self) -> str:
        return self._container.container

    def execute(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self._container.execute(
            command,
            input_text=input_text,
            check=check,
        )

    def admin_sql(self, sql: str) -> str:
        shell = r"""
exec psql \
    --username="$POSTGRES_USER" \
    --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
    --no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align
"""
        return self._container.command(
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
        return self._container.command(
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
