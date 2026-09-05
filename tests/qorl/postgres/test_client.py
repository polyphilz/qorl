import json
import subprocess
from unittest.mock import Mock

import pytest

from qorl.postgres.client import PostgresClient
from qorl.postgres.exceptions import PostgresError, QueryTimeout


@pytest.mark.parametrize("analyze", [False, True])
def test_statement_timeout_has_a_specific_error_type(analyze: bool) -> None:
    execute = Mock(
        return_value=subprocess.CompletedProcess(
            ["psql"], 1, "", "ERROR: canceling statement due to statement timeout"
        )
    )
    client = PostgresClient(execute)
    with pytest.raises(QueryTimeout) as raised:
        client.explain("SELECT 1", 5_000, analyze=analyze)
    assert raised.value.timeout_ms == 5_000
    assert client.explain_calls == 1
    assert client.explain_analyze_calls == int(analyze)


@pytest.mark.parametrize("analyze", [False, True])
def test_explain_uses_supplied_runner_and_preserves_options(analyze: bool) -> None:
    document = {"Plan": {"Node Type": "Result"}}
    execute = Mock(
        return_value=subprocess.CompletedProcess(
            ["psql"], 0, json.dumps([document]), "hint diagnostic"
        )
    )
    client = PostgresClient(execute)
    result = client.explain("SELECT 1;", 5_000, analyze=analyze, hint="/*+ hint */")
    command, sql = execute.call_args.args
    assert command[:4] == ["bash", "-Eeuo", "pipefail", "-c"]
    assert "docker" not in command
    assert "statement_timeout=5000" in command[-1]
    assert "pg_hint_plan.debug_print=detailed" in command[-1]
    options = "ANALYZE, TIMING OFF, BUFFERS, FORMAT JSON" if analyze else "FORMAT JSON"
    assert sql == f"EXPLAIN ({options})\n/*+ hint */\nSELECT 1;\n"
    assert result.document == document
    assert result.hint_diagnostics == "hint diagnostic"


@pytest.mark.parametrize("csv", [False, True])
def test_runner_output_format_is_explicit(csv: bool) -> None:
    execute = Mock(return_value=subprocess.CompletedProcess(["psql"], 0, "output", ""))
    client = PostgresClient(execute)
    assert client.runner_sql("SELECT 1", csv=csv) == "output"
    command, sql = execute.call_args.args
    assert sql == "SELECT 1"
    assert ("--csv" in command[-1]) == csv
    assert ("--tuples-only --no-align" in command[-1]) == (not csv)


def test_query_error_preserves_executed_command_and_stderr() -> None:
    command = ["docker", "exec", "--interactive", "container", "psql"]
    client = PostgresClient(
        Mock(return_value=subprocess.CompletedProcess(command, 1, "", "query error\n"))
    )
    with pytest.raises(PostgresError) as error:
        client.runner_sql("bad query")
    assert (
        str(error.value)
        == "command failed (1): docker exec --interactive container psql\nquery error"
    )
