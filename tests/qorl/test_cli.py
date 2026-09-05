from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl import cli
from qorl.cli import parser


@pytest.mark.parametrize("command", ["calibrate", "run"])
@pytest.mark.parametrize("omitted", ["--postgres-config", "--pool-config"])
def test_both_config_paths_are_required(command: str, omitted: str) -> None:
    options = {
        "--postgres-config": "docker/postgres/configs/001-pgconf",
        "--pool-config": "docker/worker_pool/configs/001-poolconf-2x16",
    }
    arguments = [command]
    for flag, path in options.items():
        if flag != omitted:
            arguments.extend([flag, path])
    with pytest.raises(SystemExit) as error:
        parser().parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["calibrate", "run"])
def test_cli_forwards_both_selections(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    postgres = Path("docker/postgres/configs/001-pgconf")
    pool = Path("docker/worker_pool/configs/001-poolconf-2x16")
    execute = Mock(return_value=tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QORL_RL_WORKER_POOL_CONFIG", "must-not-be-used")
    monkeypatch.setattr(
        cli, "calibrate" if command == "calibrate" else "run_benchmark", execute
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qorl",
            command,
            "--postgres-config",
            str(postgres),
            "--pool-config",
            str(pool),
        ],
    )
    assert cli.main() == 0
    args = (tmp_path, "job", None, None) if command == "calibrate" else (tmp_path,)
    execute.assert_called_once_with(
        *args, postgres_config_path=postgres, pool_config_path=pool
    )
