"""Dispatch experiment stages using numbered outputs and recorded inputs."""

import argparse
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import tomli_w

from qorl.evaluation.evaluate import evaluate
from qorl.experiment.schemas import (
    PLACEHOLDER,
    CalibrationExperimentConfig,
    ExperimentConfig,
    ExperimentMethod,
    ModelExperimentConfig,
    RlExperimentConfig,
    RunInputs,
    RunRequest,
    RunStage,
    SftExperimentConfig,
    load_config,
)
from qorl.measure.calibration import calibrate
from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.config import PostgresConfig
from qorl.taskset.schemas import TaskRole, TaskSelection, TaskSelectionInput
from qorl.taskset.selection import validate_splits
from qorl.taskset.taskset import TaskSet
from qorl.util.io import write_json
from qorl.worker_pool.config import load_pool_config

OUTPUTS_DIRECTORY = REPOSITORY_ROOT / "outputs"
RUN_NUMBER_WIDTH = 3
INTERRUPTED_EXIT_CODE = 130
METHOD_STAGES = {
    ExperimentMethod.SFT: (RunStage.PREPARE, RunStage.TRAIN, RunStage.EVALUATE),
    ExperimentMethod.RL: (RunStage.TRAIN, RunStage.EVALUATE),
    ExperimentMethod.EVAL: (RunStage.EVALUATE,),
    ExperimentMethod.CALIBRATE: (RunStage.CALIBRATE,),
}


def run_number(value: str) -> int:
    """Accept a nonnegative run number, including zero-padded names."""
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("run number must be a nonnegative integer")
    return int(value)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Define the same stage flags for the CLI and generated entrypoint."""
    parser.add_argument("--stage", required=True, choices=list(RunStage))
    parser.add_argument(
        "--run", type=run_number, help="existing run number; never inferred"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="saved adapter for training evaluation"
    )
    parser.add_argument("--split", choices=[TaskRole.VALIDATION, TaskRole.TEST])
    parser.add_argument(
        "--resume", action="store_true", help="continue saved stage progress"
    )


def request_from_arguments(arguments: argparse.Namespace) -> RunRequest:
    """Convert parsed arguments into the shared stage request."""
    return RunRequest(
        stage=RunStage(arguments.stage),
        number=arguments.run,
        checkpoint=arguments.checkpoint,
        split=TaskRole(arguments.split) if arguments.split is not None else None,
        resume=arguments.resume,
    )


def launch_experiment(directory: Path, request: RunRequest) -> int:
    """Execute the experiment's own entrypoint and preserve its exit status."""
    directory = (REPOSITORY_ROOT / directory).resolve()
    script = directory / "run.py"
    if not script.is_file():
        raise ValueError(f"experiment entrypoint is missing: {script}")
    arguments = ["--stage", request.stage.value]
    if request.number is not None:
        arguments.extend(["--run", str(request.number)])
    if request.checkpoint is not None:
        arguments.extend(
            ["--checkpoint", str((Path.cwd() / request.checkpoint).resolve())]
        )
    if request.split is not None:
        arguments.extend(["--split", request.split.value])
    if request.resume:
        arguments.append("--resume")
    with subprocess.Popen(
        [sys.executable, str(script), *arguments],
        cwd=REPOSITORY_ROOT,
        start_new_session=True,
    ) as process:
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            return process.wait()


def task_inputs(config: ExperimentConfig) -> dict[TaskRole, TaskSelectionInput]:
    """List only the task roles recorded by this experiment method."""
    if isinstance(config, (SftExperimentConfig, RlExperimentConfig)):
        inputs = {
            TaskRole.TRAIN: config.data.training,
            TaskRole.VALIDATION: config.data.validation,
        }
        if config.data.test is not None:
            inputs[TaskRole.TEST] = config.data.test
        return inputs
    return {TaskRole.TEST: config.data.test}


def task_file(directory: Path, path: Path) -> Path:
    """Keep task-selection files inside their owning experiment or run."""
    resolved = (directory / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"task file must be relative and inside its directory: {path}")
    return resolved


def load_inputs(directory: Path) -> RunInputs:
    """Validate saved IDs and topology separation without resampling."""
    config = load_config(directory / "config.toml")
    selections = {
        role: TaskSelection.model_validate_json(
            task_file(directory, item.file).read_bytes()
        )
        for role, item in task_inputs(config).items()
    }
    catalogs = {
        selection.benchmark_id.value: TaskSet.load(
            REPOSITORY_ROOT, selection.benchmark_id.value
        )
        for selection in selections.values()
    }
    validate_splits(selections, catalogs)
    return RunInputs(config, selections)


def validate_stage(config: ExperimentConfig, request: RunRequest) -> None:
    """Reject irrelevant flags and require an explicit run for subsequent stages."""
    method = config.experiment.method
    if request.stage not in METHOD_STAGES[method]:
        raise ValueError(f"stage {request.stage.value} is not valid for {method.value}")
    if request.number is not None and request.number < 0:
        raise ValueError("run number must be nonnegative")
    if request.stage == RunStage.EVALUATE and request.resume:
        raise ValueError("evaluation cannot resume; restart into fresh outputs")
    if request.resume and request.number is None:
        raise ValueError("--resume requires --run")
    if request.number is None and request.stage != METHOD_STAGES[method][0]:
        raise ValueError(f"stage {request.stage.value} requires --run")
    if request.stage == RunStage.EVALUATE:
        if (
            isinstance(config, ModelExperimentConfig)
            and config.agent.candidate_attempts != 1
        ):
            raise ValueError("measured evaluation requires agent.candidate_attempts=1")
        if request.split is None:
            raise ValueError("evaluation requires --split")
        if request.split not in task_inputs(config):
            raise ValueError(f"experiment has no {request.split.value} taskset")
        if method in (ExperimentMethod.SFT, ExperimentMethod.RL):
            if request.checkpoint is None:
                raise ValueError("training evaluation requires --checkpoint")
        elif request.checkpoint is not None:
            raise ValueError(
                "standalone evaluation uses its configured model and adapter"
            )
    elif request.split is not None or request.checkpoint is not None:
        raise ValueError("--split and --checkpoint apply only to evaluation")


def numbered_output(parent: Path) -> Path:
    """Claim the next numeric directory without replacing any existing entry."""
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        numbers = [
            int(path.name)
            for path in parent.iterdir()
            if path.name.isascii() and path.name.isdecimal()
        ]
        number = max(numbers, default=-1) + 1
        output = parent / f"{number:0{RUN_NUMBER_WIDTH}d}"
        try:
            output.mkdir()
            return output
        except FileExistsError:
            continue


def create_run(directory: Path, inputs: RunInputs) -> Path:
    """Allocate a new run and copy its inputs; never replace an existing run."""
    output = numbered_output(OUTPUTS_DIRECTORY / directory.name)
    try:
        (output / "config.toml").write_text(
            tomli_w.dumps(inputs.config.model_dump(mode="json", exclude_none=True)),
            encoding="utf-8",
        )
        for role, item in task_inputs(inputs.config).items():
            path = task_file(output, item.file)
            write_json(path, inputs.selections[role].model_dump(mode="json"))
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def run_experiment(directory: Path, request: RunRequest) -> Path:
    """Dispatch calibration or evaluation using recorded inputs and fresh outputs."""
    directory = (REPOSITORY_ROOT / directory).resolve()
    config = load_config(directory / "config.toml")
    validate_stage(config, request)
    if request.stage not in (RunStage.CALIBRATE, RunStage.EVALUATE):
        raise NotImplementedError(f"stage {request.stage.value} is not implemented")
    if request.resume:
        raise ValueError("calibration cannot resume partial runs; start a new run")

    current = load_inputs(directory)
    if request.number is None:
        inputs = current
        output = None
    else:
        output = (
            OUTPUTS_DIRECTORY
            / directory.name
            / f"{request.number:0{RUN_NUMBER_WIDTH}d}"
        )
        if not output.is_dir():
            raise ValueError(f"run does not exist: {output}")
        inputs = load_inputs(output)
        if inputs != current:
            raise ValueError("experiment inputs changed; start a new run without --run")
        if request.stage == RunStage.CALIBRATE and (output / "calibration").exists():
            raise ValueError(
                f"calibration output already exists; refusing to overwrite: {output}"
            )

    config = inputs.config
    if PLACEHOLDER in (str(config.postgres.path), str(config.pool.path)):
        raise ValueError("fill in the PostgreSQL and pool configuration paths")
    postgres_config = PostgresConfig.load(config.postgres.path)
    pool_config = load_pool_config(config.pool.path)
    role = request.split if request.stage == RunStage.EVALUATE else TaskRole.TEST
    if role is None:
        raise ValueError("evaluation requires --split")
    selection = inputs.selections[role]
    task_set = TaskSet.load(REPOSITORY_ROOT, selection.benchmark_id.value)

    if (
        isinstance(config, ModelExperimentConfig)
        and config.model.name_or_path == PLACEHOLDER
    ):
        raise ValueError("fill in the model identity before evaluation")

    if output is None:
        output = create_run(directory, inputs)
    print(f"QORL run {output.name}: {output}", flush=True)
    if isinstance(config, CalibrationExperimentConfig):
        calibrate(
            task_set,
            selection,
            output / "calibration",
            settings=config.measurement,
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
    else:
        model = config.model
        if request.checkpoint is not None:
            model = model.model_copy(
                update={
                    "adapter_path": (
                        REPOSITORY_ROOT / request.checkpoint.expanduser()
                    ).resolve(),
                }
            )
        # Each invocation owns a fresh slot so a restart cannot reuse partial records.
        invocation = numbered_output(output / "evaluation" / role.value)
        print(f"QORL evaluation results: {invocation}", flush=True)
        report = evaluate(
            task_set,
            selection,
            invocation,
            split=role,
            model=model,
            inference=config.inference,
            serving_gpu_ids=config.resources.serving_gpu_ids
            if config.resources
            else None,
            agent=config.agent,
            measurement=config.measurement,
            settings=config.evaluation,
            seed=config.experiment.seed,
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
        summary = report.summary
        print(
            f"Plans: {summary.valid_plan_rollout_count} valid, "
            f"{summary.novel_plan_rollout_count} novel / "
            f"{summary.recorded_rollout_count} rollouts; "
            f"{summary.distinct_novel_plan_count} distinct novel task/plans. "
            f"Speedup-bearing outcomes: {summary.performance.scored_rollout_count}; "
            f"timeouts: {summary.performance.timeout_count}.",
            flush=True,
        )
    return output


def main(experiment_directory: Path, arguments: Sequence[str] | None = None) -> int:
    """Parse a generated entrypoint's stage arguments and report its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    request = request_from_arguments(parser.parse_args(arguments))
    try:
        output = run_experiment(experiment_directory, request)
    except KeyboardInterrupt:
        print("qorl: interrupted; partial run outputs are retained", file=sys.stderr)
        return INTERRUPTED_EXIT_CODE
    except (RuntimeError, OSError, ValueError) as error:
        print(f"qorl: {error}", file=sys.stderr)
        return 1
    print(f"QORL {request.stage.value} complete: {output}")
    return 0
