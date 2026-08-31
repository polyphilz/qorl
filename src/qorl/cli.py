from __future__ import annotations

import argparse
from pathlib import Path

from qorl import __version__
from qorl.benchmark import run_benchmark
from qorl.calibration import calibrate
from qorl.rl import rl
from qorl.sft import sft


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qorl")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    commands.add_parser(
        "calibrate", help="measure PostgreSQL's default plans on JOB"
    )
    commands.add_parser("run", help="run the configured policy on JOB")
    commands.add_parser("sft", help="train the protocol-SFT LoRA adapter")
    commands.add_parser("rl", help="run reinforcement learning on CEB")
    return root


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command is None:
        parser().print_help()
        return 0
    try:
        actions = {
            "calibrate": calibrate,
            "run": run_benchmark,
            "sft": sft,
            "rl": rl,
        }
        output_dir = actions[arguments.command](Path.cwd())
    except (RuntimeError, OSError) as error:
        print(f"qorl: {error}")
        return 1
    print(f"QORL {arguments.command} complete: {output_dir}")
    return 0
