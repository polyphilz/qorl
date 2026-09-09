"""Delegate this experiment to QORL's shared stage implementation."""

from pathlib import Path

from qorl.experiment.run import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
