from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from qorl import __version__
from qorl.adapters.merge import merge
from qorl.adapters.schemas import MergeLoraConfig
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
from qorl.model.files import resolve_model
from qorl.model.schemas import ModelProvider, ModelSettings
from qorl.paths import REPOSITORY_ROOT


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    model_parser = commands.add_parser("model", help="model artifacts")
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    merge_parser = model_commands.add_parser(
        "merge", help="merge an exported LoRA adapter into its base"
    )
    merge_parser.add_argument("--adapter-path", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--base-model-name-or-path")
    merge_parser.add_argument(
        "--base-model-revision", help="already-cached Hugging Face revision"
    )
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
        "--exclude-tasks-from",
        type=Path,
        action="append",
        default=[],
        help="exclude IDs and identical SQL from a saved task-selection JSON before sampling; repeatable",
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
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    root = parser()
    arguments = root.parse_args()
    if arguments.command is None:
        root.print_help()
        return 0
    try:
        if arguments.command == "model":
            adapter = arguments.adapter_path.expanduser().resolve()
            config = MergeLoraConfig.model_validate_json(
                (adapter / "adapter_config.json").read_bytes()
            )
            base = resolve_model(
                ModelSettings(
                    provider=ModelProvider.LOCAL,
                    name_or_path=arguments.base_model_name_or_path
                    or config.base_model_name_or_path,
                    revision=arguments.base_model_revision
                    or (
                        config.revision
                        if arguments.base_model_name_or_path is None
                        else None
                    ),
                    context_length=1,
                )
            )
            print(merge(base, adapter, arguments.output.expanduser().absolute()))
            return 0
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
                    exclude_tasks_from=tuple(arguments.exclude_tasks_from),
                )
            )
            print(f"QORL experiment created: {output_dir}")
            return 0
    except (RuntimeError, OSError, ValueError) as error:
        print(f"qorl: {error}")
        return 1
    return 0
