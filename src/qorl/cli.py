from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
from qorl.db.config import DEFAULT_POSTGRES_CONFIG
from qorl.db.resources import DEFAULT_POOL_CONFIG
from qorl.evaluation.benchmark import run_benchmark
from qorl.measure.calibration import calibrate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    calibrate_parser = commands.add_parser(
        "calibrate", help="measure PostgreSQL's default plans"
    )
    calibrate_parser.add_argument(
        "workload",
        nargs="?",
        default="job",
        choices=("job", "ceb"),
        help="query workload to calibrate (default: job)",
    )
    calibrate_parser.add_argument(
        "--selection",
        type=Path,
        help="versioned task-selection manifest (default: entire workload)",
    )
    calibrate_parser.add_argument(
        "--split",
        help="selection split; inferred when the manifest has only one",
    )
    calibrate_parser.add_argument(
        "--postgres-config",
        type=Path,
        default=DEFAULT_POSTGRES_CONFIG,
        help=(f"PostgreSQL config directory (default: {DEFAULT_POSTGRES_CONFIG})"),
    )
    run_parser = commands.add_parser("run", help="run the configured policy on JOB")
    for command in (calibrate_parser, run_parser):
        command.add_argument(
            "--pool-config",
            type=Path,
            help=(
                "worker pool config directory or poolconf.json "
                f"(default: QORL_RL_WORKER_POOL_CONFIG or {DEFAULT_POOL_CONFIG})"
            ),
        )
    return root


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command is None:
        parser().print_help()
        return 0
    try:
        if arguments.command == "calibrate":
            output_dir = calibrate(
                Path.cwd(),
                arguments.workload,
                arguments.selection,
                arguments.split,
                arguments.postgres_config,
                arguments.pool_config,
            )
        else:
            output_dir = run_benchmark(
                Path.cwd(), pool_config_path=arguments.pool_config
            )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"qorl: {error}")
        return 1
    print(f"QORL {arguments.command} complete: {output_dir}")
    return 0
