#!/usr/bin/env python3
"""Confirm the prompt's live planner settings match benchmark-v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qorl.plans.action import (  # noqa: E402
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    NUMERIC_SETTINGS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "docker/postgres/contract/benchmark.expected.json",
    )
    args = parser.parse_args()

    names = sorted(
        set(BOOLEAN_SETTINGS) | set(INTEGER_SETTINGS) | set(NUMERIC_SETTINGS)
    )
    expected_settings = json.loads(args.contract.read_text(encoding="utf-8"))[
        "settings"
    ]
    expected = {name: str(expected_settings[name]) for name in names}
    literals = ", ".join("'" + name.replace("'", "''") + "'" for name in names)
    sql = (
        "SELECT json_object_agg(name, setting ORDER BY name) "
        "FROM pg_settings "
        f"WHERE name IN ({literals});"
    )
    command = [
        "docker",
        "exec",
        "--interactive",
        args.container,
        "bash",
        "-Eeuo",
        "pipefail",
        "-c",
        'exec env PGPASSWORD="$QORL_RUNNER_PASSWORD" '
        "PGAPPNAME=qorl-prompt-settings-check "
        "psql --host=127.0.0.1 --username=qorl_runner "
        '--dbname="${POSTGRES_DB:-$POSTGRES_USER}" '
        "--no-psqlrc --set=ON_ERROR_STOP=1 --quiet --tuples-only --no-align",
    ]
    completed = subprocess.run(
        command,
        input=sql,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr.strip())
    actual = json.loads(completed.stdout)

    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    actual_json = json.dumps(actual, sort_keys=True, separators=(",", ":"))
    if actual_json != expected_json:
        raise SystemExit(
            "live prompt planner settings differ from benchmark-v2\n"
            f"expected={expected_json}\nactual={actual_json}"
        )
    print("Live prompt planner settings match benchmark-v2.")


if __name__ == "__main__":
    main()
