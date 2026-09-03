from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
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
    commands.add_parser("run", help="run the configured policy on JOB")
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
            )
        else:
            output_dir = run_benchmark(Path.cwd())
    except (RuntimeError, OSError) as error:
        print(f"qorl: {error}")
        return 1
    print(f"QORL {arguments.command} complete: {output_dir}")
    return 0
