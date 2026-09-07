from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
from qorl.evaluation.benchmark import run_benchmark
from qorl.experiment.create import create_experiment
from qorl.experiment.schemas import (
    DEFAULT_EXPERIMENT_SEED,
    CreateRequest,
    ExperimentMethod,
)
from qorl.measure.calibration import (
    DEFAULT_MAX_WARMUP_RUNS,
    DEFAULT_NUM_TRIALS,
    calibrate,
    validate_run_counts,
)
from qorl.model.schemas import ModelProvider
from qorl.paths import REPOSITORY_ROOT


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    experiment_parser = commands.add_parser(
        "experiment", help="create experiment configuration and task selections"
    )
    experiment_commands = experiment_parser.add_subparsers(
        dest="experiment_command", required=True
    )
    create_parser = experiment_commands.add_parser(
        "create", help="write a new experiment; no execution"
    )
    create_parser.add_argument(
        "--name", required=True, help="short lowercase experiment name"
    )
    create_parser.add_argument(
        "--method", required=True, choices=list(ExperimentMethod)
    )
    create_parser.add_argument(
        "--base-model-name-or-path",
        help="complete local model, Hugging Face ID, or hosted model ID",
    )
    create_parser.add_argument(
        "--base-model-revision", help="immutable 40-character Hugging Face commit"
    )
    create_parser.add_argument(
        "--model-provider",
        choices=list(ModelProvider),
        default=ModelProvider.LOCAL.value,
    )
    create_parser.add_argument(
        "--adapter-path",
        type=Path,
        help="separate adapter for standalone local evaluation",
    )
    create_parser.add_argument(
        "--dataset-from", type=Path, help="QORL conversation artifact to reuse for SFT"
    )
    create_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EXPERIMENT_SEED,
        help="experiment random seed (default: %(default)s)",
    )
    create_parser.add_argument(
        "--tasksets",
        nargs="+",
        default=[],
        help="named selections, e.g. 'train=ceb[2a:20]' 'validation=ceb[4a:20]' 'test=job'",
    )
    create_parser.add_argument(
        "--postgres-config",
        type=Path,
        required=True,
        help="existing PostgreSQL config directory",
    )
    create_parser.add_argument(
        "--pool-config",
        type=Path,
        required=True,
        help="existing worker pool config directory or poolconf.json",
    )
    calibrate_parser = commands.add_parser(
        "calibrate", help="measure PostgreSQL's default plans on all JOB queries"
    )
    calibrate_parser.add_argument(
        "--max-warmup-runs",
        type=int,
        default=DEFAULT_MAX_WARMUP_RUNS,
        help="maximum warmups per query (minimum: 2; default: %(default)s)",
    )
    calibrate_parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="measured executions per query, excluding warmups (minimum: 2; default: %(default)s)",
    )
    run_parser = commands.add_parser("run", help="run the configured policy on JOB")
    for command in (calibrate_parser, run_parser):
        command.add_argument(
            "--postgres-config",
            type=Path,
            required=True,
            help="PostgreSQL config directory",
        )
        command.add_argument(
            "--pool-config",
            type=Path,
            required=True,
            help="worker pool config directory or poolconf.json",
        )
    return root


def main() -> int:
    root = parser()
    arguments = root.parse_args()
    if arguments.command is None:
        root.print_help()
        return 0
    if arguments.command == "calibrate":
        try:
            validate_run_counts(arguments.max_warmup_runs, arguments.num_trials)
        except ValueError as error:
            root.error(str(error))
    try:
        if arguments.command == "experiment":
            output_dir = create_experiment(
                CreateRequest(
                    name=arguments.name,
                    method=ExperimentMethod(arguments.method),
                    postgres_config=arguments.postgres_config,
                    pool_config=arguments.pool_config,
                    tasksets=tuple(arguments.tasksets),
                    seed=arguments.seed,
                    base_model_name_or_path=arguments.base_model_name_or_path,
                    base_model_revision=arguments.base_model_revision,
                    model_provider=ModelProvider(arguments.model_provider),
                    adapter_path=arguments.adapter_path,
                    dataset_from=arguments.dataset_from,
                )
            )
            print(f"QORL experiment created: {output_dir}")
            return 0
        elif arguments.command == "calibrate":
            output_dir = calibrate(
                REPOSITORY_ROOT,
                postgres_config_path=arguments.postgres_config,
                pool_config_path=arguments.pool_config,
                max_warmup_runs=arguments.max_warmup_runs,
                num_trials=arguments.num_trials,
            )
        else:
            output_dir = run_benchmark(
                REPOSITORY_ROOT,
                postgres_config_path=arguments.postgres_config,
                pool_config_path=arguments.pool_config,
            )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"qorl: {error}")
        return 1
    print(f"QORL {arguments.command} complete: {output_dir}")
    return 0
