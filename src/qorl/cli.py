from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
from qorl.experiment.create import create_experiment
from qorl.experiment.run import (
    add_run_arguments,
    launch_experiment,
    request_from_arguments,
)
from qorl.experiment.schemas import (
    DEFAULT_EXPERIMENT_SEED,
    CreateRequest,
    ExperimentMethod,
)
from qorl.model.schemas import ModelProvider


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    experiment_parser = commands.add_parser(
        "experiment", help="create and run experiments"
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
    experiment_run = experiment_commands.add_parser(
        "run", help="execute an experiment's run.py with an explicit stage"
    )
    experiment_run.add_argument("experiment_directory", type=Path)
    add_run_arguments(experiment_run)
    return root


def main() -> int:
    root = parser()
    arguments = root.parse_args()
    if arguments.command is None:
        root.print_help()
        return 0
    try:
        if arguments.command == "experiment":
            if arguments.experiment_command == "run":
                return launch_experiment(
                    arguments.experiment_directory, request_from_arguments(arguments)
                )
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
    except (RuntimeError, OSError, ValueError) as error:
        print(f"qorl: {error}")
        return 1
    return 0
