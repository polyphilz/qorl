from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl import cli
from qorl.cli import parser


@pytest.mark.parametrize("command", ["calibrate", "run"])
@pytest.mark.parametrize("omitted", ["--postgres-config", "--pool-config"])
def test_both_config_paths_are_required(command: str, omitted: str) -> None:
    options = {
        "--postgres-config": "docker/postgres/configs/000-pgconf-default",
        "--pool-config": "docker/worker_pool/configs/001-poolconf-2x16",
    }
    arguments = [command]
    for flag, path in options.items():
        if flag != omitted:
            arguments.extend([flag, path])
    with pytest.raises(SystemExit) as error:
        parser().parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("command", "run_flags", "max_warmup_runs", "num_trials"),
    [
        ("calibrate", [], 5, 20),
        ("calibrate", ["--max-warmup-runs", "2", "--num-trials", "2"], 2, 2),
        ("calibrate", ["--max-warmup-runs", "7", "--num-trials", "30"], 7, 30),
        ("run", [], None, None),
    ],
)
def test_cli_forwards_both_selections(
    command: str,
    run_flags: list[str],
    max_warmup_runs: int | None,
    num_trials: int | None,
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres = tmp_path / "test-pgconf"
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
            *run_flags,
            "--postgres-config",
            str(postgres),
            "--pool-config",
            str(pool),
        ],
    )
    assert cli.main() == 0
    if command == "calibrate":
        execute.assert_called_once_with(
            repository_root,
            postgres_config_path=postgres,
            pool_config_path=pool,
            max_warmup_runs=max_warmup_runs,
            num_trials=num_trials,
        )
    else:
        execute.assert_called_once_with(
            repository_root, postgres_config_path=postgres, pool_config_path=pool
        )


@pytest.mark.parametrize("flag", ["--max-warmup-runs", "--num-trials"])
@pytest.mark.parametrize("value", ["-1", "0", "1", "2.5", "invalid"])
def test_calibrate_validates_counts_before_running(
    flag: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    execute = Mock()
    monkeypatch.setattr(cli, "calibrate", execute)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qorl",
            "calibrate",
            "--postgres-config",
            "docker/postgres/configs/000-pgconf-default",
            "--pool-config",
            "docker/worker_pool/configs/002-poolconf-4x8",
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    execute.assert_not_called()
    message = capsys.readouterr().err
    assert "at least 2" in message or "invalid int value" in message
