import argparse
from pathlib import Path

from qorl.evaluation.benchmark import run_benchmark


def run_random_benchmark(
    repository: Path, *, postgres_config_path: Path, pool_config_path: Path
) -> Path:
    return run_benchmark(
        repository,
        "experiments/000-vanilla-baseline/random.json",
        postgres_config_path=postgres_config_path,
        pool_config_path=pool_config_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen random structured-action baseline."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--postgres-config", type=Path, required=True)
    parser.add_argument("--pool-config", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        run_random_benchmark(
            arguments.repository.resolve(),
            postgres_config_path=arguments.postgres_config,
            pool_config_path=arguments.pool_config,
        )
    )


if __name__ == "__main__":
    main()
