import argparse
from pathlib import Path

from qorl.evaluation.benchmark import run_benchmark


def run_random_benchmark(repository: Path) -> Path:
    return run_benchmark(repository, "experiments/000-vanilla-baseline/random.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen random structured-action baseline."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    print(run_random_benchmark(arguments.repository.resolve()))


if __name__ == "__main__":
    main()
