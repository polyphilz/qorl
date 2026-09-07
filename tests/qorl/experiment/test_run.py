import runpy
import signal
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import tomli_w

from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CalibrationExperimentConfig,
    CreateRequest,
    ExperimentMethod,
    RunRequest,
    RunStage,
    load_config,
)
from qorl.measure.schemas import CalibrationSettings
from qorl.postgres.config import PostgresConfig
from qorl.taskset.schemas import TaskRole, TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig


@pytest.fixture
def calibration_experiment(
    creation_request: CreateRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", tmp_path / "outputs")
    return create.create_experiment(
        replace(
            creation_request,
            name="selected-calibration",
            method=ExperimentMethod.CALIBRATE,
            base_model_name_or_path=None,
            base_model_revision=None,
            tasksets=("test=ceb[2a:2]",),
        )
    )


@pytest.fixture
def measure(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def execute(
        task_set: TaskSet,
        selection: TaskSelection,
        output: Path,
        *,
        settings: CalibrationSettings,
        postgres_config: PostgresConfig,
        pool_config: PoolConfig,
    ) -> None:
        output.mkdir()

    mocked = Mock(side_effect=execute)
    monkeypatch.setattr(run, "calibrate", mocked)
    return mocked


def test_new_runs_copy_inputs_and_never_overwrite_previous_results(
    calibration_experiment: Path, measure: Mock
) -> None:
    expected = run.load_inputs(calibration_experiment)
    first = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert first.name == "000"
    assert run.load_inputs(first) == expected
    assert {path.name for path in first.iterdir()} == {
        "config.toml",
        "test-tasks.json",
        "calibration",
    }
    marker = first / "calibration/result"
    marker.write_text("original result")
    second = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert second.name == "001"
    assert run.load_inputs(second) == expected
    assert marker.read_text() == "original result"
    assert measure.call_count == 2
    for call in measure.call_args_list:
        task_set, selected, output = call.args
        assert task_set.task_set_id == "ceb"
        assert selected == expected.selections[TaskRole.TEST]
        assert output.name == "calibration"
        assert call.kwargs["settings"] == expected.config.measurement


def test_run_number_uses_highest_numeric_name(
    calibration_experiment: Path, measure: Mock
) -> None:
    parent = run.OUTPUTS_DIRECTORY / calibration_experiment.name
    parent.mkdir(parents=True)
    (parent / "010").mkdir()
    (parent / "009").mkdir()
    (parent / "notes").mkdir()
    (parent / "011").write_text("not a run; still must not overwrite")
    output = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert output.name == "012"
    assert (parent / "011").read_text() == "not a run; still must not overwrite"


@pytest.mark.parametrize("changed", ["config", "tasks"])
def test_changed_experiment_inputs_require_a_new_run(
    changed: str, calibration_experiment: Path, measure: Mock
) -> None:
    first = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    saved = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
    if changed == "config":
        config = load_config(calibration_experiment / "config.toml")
        updated = config.model_copy(
            update={"experiment": config.experiment.model_copy(update={"seed": 123})}
        )
        (calibration_experiment / "config.toml").write_text(
            tomli_w.dumps(updated.model_dump(mode="json", exclude_none=True))
        )
    else:
        path = calibration_experiment / "test-tasks.json"
        selected = TaskSelection.model_validate_json(path.read_bytes())
        path.write_text(
            selected.model_copy(
                update={"task_ids": selected.task_ids[:1]}
            ).model_dump_json()
        )
    with pytest.raises(ValueError, match="inputs changed"):
        run.run_experiment(
            calibration_experiment, RunRequest(RunStage.CALIBRATE, number=0)
        )
    assert measure.call_count == 1
    assert {
        path.name: path.read_bytes() for path in first.iterdir() if path.is_file()
    } == saved
    assert (
        run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE)).name
        == "001"
    )


def test_existing_calibration_results_are_protected(
    calibration_experiment: Path, measure: Mock
) -> None:
    run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        run.run_experiment(
            calibration_experiment, RunRequest(RunStage.CALIBRATE, number=0)
        )
    assert measure.call_count == 1


def test_existing_empty_stage_uses_its_recorded_inputs(
    calibration_experiment: Path, measure: Mock
) -> None:
    inputs = run.load_inputs(calibration_experiment)
    output = run.create_run(calibration_experiment, inputs)
    assert (
        run.run_experiment(
            calibration_experiment, RunRequest(RunStage.CALIBRATE, number=0)
        )
        == output
    )
    measure.assert_called_once()


def test_missing_run_is_not_created(
    calibration_experiment: Path, measure: Mock
) -> None:
    with pytest.raises(ValueError, match="run does not exist"):
        run.run_experiment(
            calibration_experiment, RunRequest(RunStage.CALIBRATE, number=7)
        )
    assert not run.OUTPUTS_DIRECTORY.exists()
    measure.assert_not_called()


def test_calibration_resume_is_explicitly_unsupported_and_preserves_progress(
    calibration_experiment: Path, measure: Mock
) -> None:
    output = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    progress = output / "calibration/partial"
    progress.write_text("keep")
    with pytest.raises(ValueError, match="cannot resume"):
        run.run_experiment(
            calibration_experiment,
            RunRequest(RunStage.CALIBRATE, number=0, resume=True),
        )
    assert progress.read_text() == "keep"
    assert measure.call_count == 1


def test_changed_defaults_are_not_read_during_execution(
    calibration_experiment: Path,
    measure: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", tmp_path / "missing-defaults")
    output = run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert output.name == "000"


@pytest.mark.parametrize(
    "stage,method",
    [
        (RunStage.PREPARE, ExperimentMethod.SFT),
        (RunStage.TRAIN, ExperimentMethod.RL),
        (RunStage.EVALUATE, ExperimentMethod.EVAL),
    ],
)
def test_unimplemented_stages_do_not_allocate_outputs(
    stage: RunStage,
    method: ExperimentMethod,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "experiment"
    directory.mkdir()
    (directory / "config.toml").write_bytes(create.latest_template(method).read_bytes())
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", tmp_path / "outputs")
    request = RunRequest(
        stage, split=TaskRole.TEST if stage == RunStage.EVALUATE else None
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        run.run_experiment(directory, request)
    assert not run.OUTPUTS_DIRECTORY.exists()


@pytest.mark.parametrize(
    "method,stage_request,message",
    [
        (ExperimentMethod.CALIBRATE, RunRequest(RunStage.TRAIN), "not valid"),
        (ExperimentMethod.SFT, RunRequest(RunStage.TRAIN), "requires --run"),
        (
            ExperimentMethod.RL,
            RunRequest(RunStage.TRAIN, resume=True),
            "--resume requires --run",
        ),
        (
            ExperimentMethod.CALIBRATE,
            RunRequest(RunStage.CALIBRATE, number=-1),
            "nonnegative",
        ),
        (
            ExperimentMethod.CALIBRATE,
            RunRequest(RunStage.CALIBRATE, split=TaskRole.TEST),
            "only to evaluation",
        ),
        (
            ExperimentMethod.CALIBRATE,
            RunRequest(RunStage.CALIBRATE, checkpoint=Path("adapter")),
            "only to evaluation",
        ),
        (ExperimentMethod.EVAL, RunRequest(RunStage.EVALUATE), "requires --split"),
        (
            ExperimentMethod.EVAL,
            RunRequest(RunStage.EVALUATE, split=TaskRole.VALIDATION),
            "no validation taskset",
        ),
        (
            ExperimentMethod.EVAL,
            RunRequest(
                RunStage.EVALUATE, split=TaskRole.TEST, checkpoint=Path("adapter")
            ),
            "configured model",
        ),
        (
            ExperimentMethod.RL,
            RunRequest(RunStage.EVALUATE, number=0, split=TaskRole.VALIDATION),
            "requires --checkpoint",
        ),
    ],
)
def test_stage_applicability_and_explicit_selection(
    method: ExperimentMethod, stage_request: RunRequest, message: str
) -> None:
    config = load_config(create.latest_template(method))
    with pytest.raises(ValueError, match=message):
        run.validate_stage(config, stage_request)


@pytest.mark.parametrize(
    "method,stage_request",
    [
        (ExperimentMethod.SFT, RunRequest(RunStage.TRAIN, number=0)),
        (
            ExperimentMethod.SFT,
            RunRequest(
                RunStage.EVALUATE,
                number=0,
                split=TaskRole.VALIDATION,
                checkpoint=Path("adapter"),
            ),
        ),
        (ExperimentMethod.RL, RunRequest(RunStage.TRAIN)),
        (ExperimentMethod.EVAL, RunRequest(RunStage.EVALUATE, split=TaskRole.TEST)),
        (ExperimentMethod.CALIBRATE, RunRequest(RunStage.CALIBRATE)),
    ],
)
def test_valid_stage_requests(
    method: ExperimentMethod, stage_request: RunRequest
) -> None:
    run.validate_stage(load_config(create.latest_template(method)), stage_request)


def test_invalid_selection_prevents_run_creation(
    calibration_experiment: Path, measure: Mock
) -> None:
    path = calibration_experiment / "test-tasks.json"
    selected = TaskSelection.model_validate_json(path.read_bytes())
    path.write_text(
        selected.model_copy(
            update={"task_ids": ["ceb-does-not-exist"]}
        ).model_dump_json()
    )
    with pytest.raises(RuntimeError, match="unknown selected task"):
        run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert not run.OUTPUTS_DIRECTORY.exists()
    measure.assert_not_called()


def test_missing_postgres_config_prevents_run_creation(
    calibration_experiment: Path, measure: Mock
) -> None:
    path = calibration_experiment / "config.toml"
    path.write_text(path.read_text().replace("000-pgconf-default", "missing-config"))
    with pytest.raises((RuntimeError, ValueError, OSError)):
        run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert not run.OUTPUTS_DIRECTORY.exists()
    measure.assert_not_called()


def test_selection_paths_cannot_escape_the_experiment(
    calibration_experiment: Path, measure: Mock
) -> None:
    config = load_config(calibration_experiment / "config.toml")
    assert isinstance(config, CalibrationExperimentConfig)
    updated = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "test": config.data.test.model_copy(
                        update={"file": Path("../outside.json")}
                    )
                }
            )
        }
    )
    (calibration_experiment / "config.toml").write_text(
        tomli_w.dumps(updated.model_dump(mode="json", exclude_none=True))
    )
    with pytest.raises(ValueError, match="inside its directory"):
        run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert not run.OUTPUTS_DIRECTORY.exists()


def test_input_copy_failure_removes_only_the_new_run(
    calibration_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = run.OUTPUTS_DIRECTORY / calibration_experiment.name
    previous = parent / "000"
    previous.mkdir(parents=True)
    (previous / "result").write_text("keep")
    monkeypatch.setattr(run, "write_json", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        run.run_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
    assert list(parent.iterdir()) == [previous]
    assert (previous / "result").read_text() == "keep"


def test_launch_executes_the_experiment_entrypoint_and_forwards_flags(
    calibration_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.return_value = 7
    execute = Mock(return_value=process)
    monkeypatch.setattr(run.subprocess, "Popen", execute)
    checkpoint = calibration_experiment / "checkpoint"
    request = RunRequest(
        RunStage.EVALUATE,
        number=3,
        split=TaskRole.VALIDATION,
        checkpoint=checkpoint,
        resume=True,
    )
    assert run.launch_experiment(calibration_experiment, request) == 7
    execute.assert_called_once_with(
        [
            sys.executable,
            str(calibration_experiment / "run.py"),
            "--stage",
            "evaluate",
            "--run",
            "3",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "validation",
            "--resume",
        ],
        cwd=run.REPOSITORY_ROOT,
        start_new_session=True,
    )


def test_launch_forwards_interrupt_and_waits_for_child_cleanup(
    calibration_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.side_effect = [KeyboardInterrupt(), run.INTERRUPTED_EXIT_CODE]
    monkeypatch.setattr(run.subprocess, "Popen", Mock(return_value=process))
    assert (
        run.launch_experiment(calibration_experiment, RunRequest(RunStage.CALIBRATE))
        == run.INTERRUPTED_EXIT_CODE
    )
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert process.wait.call_count == 2
    process.kill.assert_not_called()


def test_generated_script_reaches_shared_calibration(
    calibration_experiment: Path, measure: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", [str(calibration_experiment / "run.py"), "--stage", "calibrate"]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(calibration_experiment / "run.py"), run_name="__main__")
    assert error.value.code == 0
    measure.assert_called_once()
    assert (
        run.OUTPUTS_DIRECTORY / calibration_experiment.name / "000/calibration"
    ).is_dir()


def test_failed_stage_keeps_recorded_inputs_and_returns_failure(
    calibration_experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run, "calibrate", Mock(side_effect=RuntimeError("database failed"))
    )
    assert run.main(calibration_experiment, ["--stage", "calibrate"]) == 1
    assert "database failed" in capsys.readouterr().err
    output = run.OUTPUTS_DIRECTORY / calibration_experiment.name / "000"
    assert run.load_inputs(output) == run.load_inputs(calibration_experiment)


def test_interrupted_stage_keeps_inputs_and_returns_interrupted_exit_code(
    calibration_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "calibrate", Mock(side_effect=KeyboardInterrupt()))
    assert (
        run.main(calibration_experiment, ["--stage", "calibrate"])
        == run.INTERRUPTED_EXIT_CODE
    )
    assert (
        run.OUTPUTS_DIRECTORY / calibration_experiment.name / "000/config.toml"
    ).is_file()
