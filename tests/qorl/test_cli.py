from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl import cli
from qorl.cli import parser
from qorl.experiment import create
from qorl.experiment.schemas import (
    CalibrationExperimentConfig,
    RunRequest,
    RunStage,
    load_config,
)
from qorl.taskset.schemas import TaskRole


def test_experiment_run_cli_forwards_the_stage_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = Mock(return_value=7)
    monkeypatch.setattr(cli, "launch_experiment", execute)
    checkpoint = tmp_path / "adapter"
    monkeypatch.setattr(
        "sys.argv",
        [
            "qorl",
            "experiment",
            "run",
            str(tmp_path),
            "--stage",
            "evaluate",
            "--run",
            "003",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "validation",
            "--resume",
        ],
    )
    assert cli.main() == 7
    execute.assert_called_once_with(
        tmp_path,
        RunRequest(
            RunStage.EVALUATE,
            number=3,
            checkpoint=checkpoint,
            split=TaskRole.VALIDATION,
            resume=True,
        ),
    )


@pytest.mark.parametrize("number", ["-1", "1.5", "latest", ""])
def test_run_numbers_are_explicit_nonnegative_integers(number: str) -> None:
    with pytest.raises(SystemExit) as error:
        parser().parse_args(
            [
                "experiment",
                "run",
                "experiments/example",
                "--stage",
                "calibrate",
                "--run",
                number,
            ]
        )
    assert error.value.code == 2


def test_running_requires_an_explicit_stage() -> None:
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["experiment", "run", "experiments/example"])
    assert error.value.code == 2


def test_experiment_create_cli_writes_files_without_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", tmp_path / "experiments")
    execute = Mock(side_effect=AssertionError("creation executed calibration"))
    monkeypatch.setattr("qorl.experiment.run.calibrate", execute)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qorl",
            "experiment",
            "create",
            "--name",
            "small-calibration",
            "--method",
            "calibrate",
            "--tasksets",
            "test=ceb[2a:1]",
            "--seed",
            "123",
            "--postgres-config",
            "docker/postgres/configs/000-pgconf-default",
            "--pool-config",
            "docker/worker_pool/configs/002-poolconf-4x8",
        ],
    )
    assert cli.main() == 0
    directory = tmp_path / "experiments/000-small-calibration"
    config = load_config(directory / "config.toml")
    assert isinstance(config, CalibrationExperimentConfig)
    assert config.experiment.seed == 123
    assert "experiment created" in capsys.readouterr().out
    execute.assert_not_called()


@pytest.mark.parametrize(
    "flag", ["--name", "--method", "--postgres-config", "--pool-config"]
)
def test_creation_required_flags(flag: str) -> None:
    options = {
        "--name": "example",
        "--method": "calibrate",
        "--postgres-config": "pg",
        "--pool-config": "pool",
    }
    arguments = ["experiment", "create"]
    for key, value in options.items():
        if key != flag:
            arguments.extend([key, value])
    with pytest.raises(SystemExit) as error:
        parser().parse_args(arguments)
    assert error.value.code == 2


def test_creation_seed_defaults_to_42() -> None:
    arguments = parser().parse_args(
        [
            "experiment",
            "create",
            "--name",
            "example",
            "--method",
            "calibrate",
            "--postgres-config",
            "pg",
            "--pool-config",
            "pool",
            "--tasksets",
            "test=job",
        ]
    )
    assert arguments.seed == 42
