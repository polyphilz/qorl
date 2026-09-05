"""Restore the prepared IMDb archive and compare it with the loaded database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qorl.db.config import PostgresConfig
from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import load_runtime_profile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.imdb.verify import verify  # noqa: E402

POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")
POOL_CONFIG = Path("docker/worker_pool/configs/000-poolconf-1x32")


def restore_and_verify(repository: Path) -> None:
    repository = repository.resolve()
    fixture = DatabaseFixture.load(repository)
    original = repository / "data/imdb-verification/loaded.json"
    report = original.with_name("restored.json")
    if not original.is_file():
        raise RuntimeError(f"loaded database verification is missing: {original}")
    if report.exists():
        raise RuntimeError(f"refusing to overwrite verification: {report}")
    profile = load_runtime_profile(repository, POOL_CONFIG)
    config = PostgresConfig.load(repository, POSTGRES_CONFIG)
    container = PostgresContainer(
        fixture,
        "qorl-imdb-restore",
        profile,
        profile.workers[0],
        config,
    )
    try:
        container.create()
        container.restore_archive()
        container.start()
        verify(container.container, report, repository=repository, compare_to=original)
    except BaseException:
        print(f"IMDb restore failed; resources retained: {container.project_name}")
        raise
    container.close()
    print(f"IMDb restore verification passed: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    restore_and_verify(REPOSITORY_ROOT)


if __name__ == "__main__":
    main()
