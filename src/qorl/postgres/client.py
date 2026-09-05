from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable
from typing import Any

from qorl.postgres.exceptions import PostgresError, QueryTimeout
from qorl.postgres.schemas import ExplainResult


class PostgresClient:
    """Execute inspected and measured SQL using a supplied command runner."""

    def __init__(
        self,
        run_command: Callable[
            [list[str], str | None], subprocess.CompletedProcess[str]
        ],
    ) -> None:
        self._run_command = run_command
        self.explain_calls = 0
        self.explain_analyze_calls = 0
        self._settings_cache: dict[str, str] = {}

    def execute(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = self._run_command(command, input_text)
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PostgresError(
                f"command failed ({completed.returncode}): "
                f"{' '.join(completed.args)}\n{detail}"
            )
        return completed

    def admin_sql(self, sql: str) -> str:
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
            input_text=sql,
        ).stdout

    def runner_sql(
        self, sql: str, *, csv: bool = False, application_name: str = "qorl-worker"
    ) -> str:
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
        shell = shell.replace(
            "PGAPPNAME=qorl-worker", f"PGAPPNAME={shlex.quote(application_name)}"
        )
        if csv:
            shell = shell.replace("--tuples-only --no-align", "--csv")
        return self.execute(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                shell,
            ],
            input_text=sql,
        ).stdout

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
            raise PostgresError("PostgreSQL planner-setting response is incomplete")
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
        except PostgresError as error:
            if "canceling statement due to statement timeout" in str(error):
                raise QueryTimeout(timeout_ms) from error
            raise
        try:
            parsed = json.loads(completed.stdout)
            return ExplainResult(parsed[0], completed.stderr)
        except (json.JSONDecodeError, IndexError, TypeError) as error:
            raise PostgresError("PostgreSQL returned invalid EXPLAIN JSON") from error

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
