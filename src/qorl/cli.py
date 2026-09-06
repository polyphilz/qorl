from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
from qorl.evaluation.benchmark import run_benchmark
from qorl.measure.calibration import (
    DEFAULT_MAX_WARMUP_RUNS,
    DEFAULT_NUM_TRIALS,
    calibrate,
    validate_run_counts,
)
from qorl.paths import REPOSITORY_ROOT


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
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
        if arguments.command == "calibrate":
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
