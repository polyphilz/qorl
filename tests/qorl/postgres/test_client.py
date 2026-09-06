import json
import shlex
import subprocess
from unittest.mock import Mock

import pytest

from qorl.postgres.client import PostgresClient
from qorl.postgres.exceptions import PostgresError, QueryTimeout
from qorl.postgres.schemas import PostgresIndexes, PostgresSettings


@pytest.mark.parametrize("empty", [False, True])
def test_read_indexes_loads_all_public_tables(
    postgres_settings: PostgresSettings, postgres_indexes: PostgresIndexes, empty: bool
) -> None:
    expected = PostgresIndexes(by_table={}) if empty else postgres_indexes
    execute = Mock(
        return_value=subprocess.CompletedProcess(
            ["psql"], 0, expected.model_dump_json(), ""
        )
    )
    client = PostgresClient(execute, postgres_settings, postgres_indexes)

    assert client.read_indexes() == expected
    execute.assert_called_once()
    _, query = execute.call_args.args
    assert "FROM pg_indexes WHERE schemaname = 'public' GROUP BY tablename" in query
    assert "tablename IN" not in query
    assert "COALESCE" in query


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "null",
        '{"by_table": {"title": [1]}}',
        '{"by_table": {"title": "title_pkey"}}',
    ],
)
def test_read_indexes_rejects_invalid_metadata(
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
    response: str,
) -> None:
    execute = Mock(return_value=subprocess.CompletedProcess(["psql"], 0, response, ""))
    client = PostgresClient(execute, postgres_settings, postgres_indexes)

    with pytest.raises(PostgresError, match="invalid index metadata"):
        client.read_indexes()


def test_settings_are_supplied_without_a_database_query(
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    execute = Mock()
    client = PostgresClient(execute, postgres_settings, postgres_indexes)
    assert client.settings is postgres_settings
    assert client.indexes is postgres_indexes
    execute.assert_not_called()


def test_execute_passes_query_to_supplied_runner(
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    command = ["psql", "--quiet"]
    completed = subprocess.CompletedProcess(command, 0, "output", "")

    def run_command(
        arguments: list[str], query: str
    ) -> subprocess.CompletedProcess[str]:
        assert arguments == command
        assert query == "SELECT 1;"
        return completed

    client = PostgresClient(run_command, postgres_settings, postgres_indexes)
    assert client.execute(command, query="SELECT 1;") is completed


def test_admin_sql_passes_query_to_supplied_runner(
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    execute = Mock(return_value=subprocess.CompletedProcess(["psql"], 0, "output", ""))
    client = PostgresClient(execute, postgres_settings, postgres_indexes)
    assert client.admin_sql("SELECT 1;") == "output"
    command, query = execute.call_args.args
    assert command[:4] == ["bash", "-Eeuo", "pipefail", "-c"]
    assert query == "SELECT 1;"


@pytest.mark.parametrize("analyze", [False, True])
def test_statement_timeout_has_a_specific_error_type(
    analyze: bool,
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    execute = Mock(
        return_value=subprocess.CompletedProcess(
            ["psql"], 1, "", "ERROR: canceling statement due to statement timeout"
        )
    )
    client = PostgresClient(execute, postgres_settings, postgres_indexes)
    with pytest.raises(QueryTimeout) as raised:
        client.explain("SELECT 1", 5_000, analyze=analyze)
    assert raised.value.timeout_ms == 5_000
    assert client.explain_calls == 1
    assert client.explain_analyze_calls == int(analyze)


@pytest.mark.parametrize("analyze", [False, True])
@pytest.mark.parametrize("hint", ["", "/*+ hint */"])
def test_explain_uses_supplied_runner_and_preserves_options(
    analyze: bool,
    hint: str,
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    document = {"Plan": {"Node Type": "Result"}}
    execute = Mock(
        return_value=subprocess.CompletedProcess(
            ["psql"], 0, json.dumps([document]), "hint diagnostic"
        )
    )
    client = PostgresClient(execute, postgres_settings, postgres_indexes)
    result = client.explain("SELECT 1;", 5_000, analyze=analyze, hint=hint)
    command, sql = execute.call_args.args
    assert command[:4] == ["bash", "-Eeuo", "pipefail", "-c"]
    assert "docker" not in command
    arguments = shlex.split(command[-1])
    assert "--username=qorl_runner" in arguments
    assert "PGAPPNAME=qorl-worker" in arguments
    postgres_options = "PGOPTIONS=-c statement_timeout=5000"
    if hint:
        postgres_options += (
            " -c pg_hint_plan.debug_print=detailed -c pg_hint_plan.message_level=notice"
        )
    assert postgres_options in arguments
    assert "--tuples-only" in arguments
    assert "--no-align" in arguments
    assert "--csv" not in arguments
    options = "ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON" if analyze else "FORMAT JSON"
    hint_line = f"{hint}\n" if hint else ""
    assert sql == f"EXPLAIN ({options})\n{hint_line}SELECT 1;\n"
    assert result.document == document
    assert result.hint_diagnostics == "hint diagnostic"


@pytest.mark.parametrize(
    "connection_label", ["qorl-worker", "IMDb verifier's connection"]
)
def test_runner_returns_csv(
    postgres_settings: PostgresSettings,
    connection_label: str,
    postgres_indexes: PostgresIndexes,
) -> None:
    execute = Mock(return_value=subprocess.CompletedProcess(["psql"], 0, "output", ""))
    client = PostgresClient(execute, postgres_settings, postgres_indexes)
    assert client.runner_sql("SELECT 1", connection_label=connection_label) == "output"
    command, sql = execute.call_args.args
    assert sql == "SELECT 1"
    arguments = shlex.split(command[-1])
    assert "--username=qorl_runner" in arguments
    assert f"PGAPPNAME={connection_label}" in arguments
    assert "--csv" in arguments
    assert "--tuples-only" not in arguments
    assert "--no-align" not in arguments
    assert not any(argument.startswith("PGOPTIONS=") for argument in arguments)


def test_query_error_preserves_executed_command_and_stderr(
    postgres_settings: PostgresSettings,
    postgres_indexes: PostgresIndexes,
) -> None:
    command = ["docker", "exec", "--interactive", "container", "psql"]
    client = PostgresClient(
        Mock(return_value=subprocess.CompletedProcess(command, 1, "", "query error\n")),
        postgres_settings,
        postgres_indexes,
    )
    with pytest.raises(PostgresError) as error:
        client.runner_sql("bad query")
    assert (
        str(error.value)
        == "command failed (1): docker exec --interactive container psql\nquery error"
    )
