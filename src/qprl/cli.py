from __future__ import annotations

import argparse
from pathlib import Path

from qprl import __version__
from qprl.benchmark import run_random_benchmark
from qprl.calibration import calibrate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qprl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    commands.add_parser(
        "calibrate", help="measure PostgreSQL's default plans on JOB"
    )
    commands.add_parser("run", help="run the configured policy on JOB")
    return root


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command is None:
        parser().print_help()
        return 0
    try:
        output_dir = (
            calibrate(Path.cwd())
            if arguments.command == "calibrate"
            else run_random_benchmark(Path.cwd())
        )
    except (RuntimeError, OSError) as error:
        print(f"qprl: {error}")
        return 1
    print(f"QPRL {arguments.command} complete: {output_dir}")
    return 0
