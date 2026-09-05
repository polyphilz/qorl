"""Load fetched IMDb inputs, verify the database, and archive its stopped volume."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qorl.db.config import PostgresConfig
from qorl.db.container import PostgresContainer
from qorl.db.fixture import IMDB_ARCHIVE, DatabaseFixture
from qorl.db.resources import load_runtime_profile
from qorl.db.worker import PostgresWorker

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.imdb.fetch_imdb import verify_inputs  # noqa: E402
from scripts.imdb.verify import verify  # noqa: E402

POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")
POOL_CONFIG = Path("docker/worker_pool/configs/000-poolconf-1x32")
POSTGRES_UID = 999
POSTGRES_GID = 999


def archive_database(container: PostgresContainer, archive: Path) -> None:
    partial = archive.with_name(f".{archive.name}.part")
    if archive.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite an archive: {archive}")
    state = container.command(
        [
            "docker",
            "inspect",
            container.container,
            "--format",
            "{{.State.Running}} {{.State.ExitCode}}",
        ]
    ).strip()
    if state != "false 0":
        raise RuntimeError("PostgreSQL must be stopped cleanly before archiving")
    archive.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
partial=/output/{partial.name}
source=/source/$1
trap 'chown {os.getuid()}:{os.getgid()} "$partial" 2>/dev/null || true' EXIT
test -f "$source/PG_VERSION"
test ! -e "$source/postmaster.pid"
tar --create --directory="$source" --sort=name --mtime=@0 \
    --owner={POSTGRES_UID} --group={POSTGRES_GID} --numeric-owner --format=gnu . \
    | gzip --no-name --fast > "$partial"
test -s "$partial"
"""
    try:
        container.command(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--volume",
                f"{container.volume}:/source:ro",
                "--volume",
                f"{archive.parent}:/output",
                "--entrypoint",
                "bash",
                container.image_id,
                "-Eeuo",
                "pipefail",
                "-c",
                script,
                "qorl-archive",
                container.pgdata_relative_path,
            ]
        )
        partial.replace(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def load_verify_archive(repository: Path) -> None:
    repository = repository.resolve()
    archive = repository / IMDB_ARCHIVE
    report = repository / "data/imdb-verification/loaded.json"
    for path in (archive, archive.with_name(f".{archive.name}.part"), report):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    if not (repository / "data/raw/tables").is_dir():
        raise RuntimeError("IMDb CSVs are missing; run fetch_imdb first")
    verify_inputs(repository)
    profile = load_runtime_profile(repository, POOL_CONFIG)
    config = PostgresConfig.load(repository, POSTGRES_CONFIG)
    container = PostgresContainer(
        DatabaseFixture(repository, archive),
        "qorl-imdb-load",
        profile,
        profile.workers[0],
        config,
    )
    try:
        container.create()
        container.start()
        print("Loading IMDb rows, indexes, and statistics...")
        PostgresWorker(container).admin_sql(
            (repository / "scripts/imdb/load.sql").read_text(encoding="utf-8")
        )
        verify(container.container, report, repository=repository)
        container.stop()
        archive_database(container, archive)
    except BaseException:
        print(f"IMDb load failed; resources retained: {container.project_name}")
        raise
    container.close()
    print(f"IMDb loaded, verified, and archived: {archive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    load_verify_archive(REPOSITORY_ROOT)


if __name__ == "__main__":
    main()
