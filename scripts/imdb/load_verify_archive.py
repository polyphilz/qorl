"""Load fetched IMDb inputs, verify the database, and archive its stopped volume."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.util.hashing import sha256_bytes, sha256_file
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import WorkerSlot
from scripts.imdb.schemas import (
    DatabaseChecksums,
    DatabaseSnapshot,
    ImdbManifest,
    ImdbRecord,
    LoadVerificationReport,
    RepresentativeQueryOutput,
)
from scripts.shared.verify import verify_file

IMDB_ARCHIVE = Path("data/imdb.tar.gz")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")
POOL_CONFIG = Path("docker/worker_pool/configs/000-poolconf-1x32")
POSTGRES_UID = 999
POSTGRES_GID = 999
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
MAX_FRESHLY_FROZEN_XID_AGE = 1_000
JOB_REPRESENTATIVE_QUERY_1 = "1a.sql"
JOB_REPRESENTATIVE_QUERY_2 = "17b.sql"
JOB_REPRESENTATIVE_QUERY_3 = "33c.sql"


def verify_input_csvs_against_manifest(
    repository: Path,
    manifest: ImdbManifest,
) -> None:
    """Check extracted file names, byte sizes, and SHA-256 hashes against the manifest."""
    target = repository / "data/raw/tables"
    members = manifest.dataset.members
    actual_names = {path.name for path in target.iterdir() if path.is_file()}
    expected_names = set(members)
    if actual_names != expected_names:
        raise RuntimeError(
            f"extracted dataset file mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name, spec in sorted(members.items()):
        verify_file(target / name, spec.bytes, spec.sha256)


def section_checksum(
    value: JsonValue
    | ImdbRecord
    | Sequence[JsonValue | ImdbRecord]
    | Mapping[str, JsonValue | ImdbRecord],
) -> str:
    """Hash canonical UTF-8 JSON, serializing IMDb records directly."""
    encoded = json.dumps(
        value,
        default=ImdbRecord.model_dump,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sql_literal(value: str) -> str:
    """Quote and escape a string for use as a PostgreSQL literal."""
    return "'" + value.replace("'", "''") + "'"


def load_database_snapshot(
    client: PostgresClient,
    table_names: list[str],
) -> DatabaseSnapshot:
    """Capture the database's identity, structure, row counts, and statistics."""
    for table_name in table_names:
        if not IDENTIFIER.fullmatch(table_name):
            raise RuntimeError(f"unsafe table name in manifest: {table_name}")

    table_array = ", ".join(sql_literal(name) for name in table_names)
    row_queries = "\nUNION ALL\n".join(
        f"SELECT {sql_literal(name)} AS table_name, count(*)::bigint AS row_count "
        f"FROM public.{name}"
        for name in table_names
    )

    sql = Path(__file__).with_name("get_snapshot.sql").read_text(encoding="utf-8")
    sql = sql.format(table_array=table_array, row_queries=row_queries)
    return DatabaseSnapshot.model_validate_json(client.admin_sql(sql))


def validate_database_snapshot(state: DatabaseSnapshot, manifest: ImdbManifest) -> None:
    """Check expected database contents, index readiness, statistics, and vacuum state."""
    expected_rows: dict[str, int] = {}
    for member in manifest.dataset.members.values():
        if member.table is not None:
            if member.rows is None:
                raise RuntimeError(f"manifest is missing row count for {member.table}")
            expected_rows[member.table] = member.rows
    if state.table_names != sorted(expected_rows):
        raise RuntimeError(
            f"IMDb table-set mismatch: expected={sorted(expected_rows)} "
            f"actual={state.table_names}"
        )
    if state.table_rows != expected_rows:
        mismatches = {
            name: {"expected": expected_rows.get(name), "actual": actual}
            for name, actual in state.table_rows.items()
            if expected_rows.get(name) != actual
        }
        raise RuntimeError(f"IMDb row-count mismatch: {mismatches}")

    identity = state.identity
    database = manifest.database
    for key, expected, actual in (
        (
            "server_version_num",
            database.server_version_num,
            identity.server_version_num,
        ),
        ("encoding", database.encoding, identity.encoding),
        ("collation", database.collation, identity.collation),
        ("ctype", database.ctype, identity.ctype),
    ):
        if actual != expected:
            raise RuntimeError(
                f"database identity mismatch for {key}: "
                f"expected={expected} actual={actual}"
            )

    indexes = state.indexes or []
    primary_count = sum(index.primary for index in indexes)
    secondary_count = sum(not index.primary for index in indexes)
    invalid = [index.name for index in indexes if not index.valid or not index.ready]
    if len(indexes) != database.expected_total_index_count:
        raise RuntimeError(f"unexpected total index count: {len(indexes)}")
    if primary_count != database.expected_primary_key_count:
        raise RuntimeError(f"unexpected primary-key index count: {primary_count}")
    if secondary_count != database.expected_secondary_index_count:
        raise RuntimeError(f"unexpected secondary index count: {secondary_count}")
    if invalid:
        raise RuntimeError(f"invalid or unready indexes: {invalid}")

    statistics_tables = {row.table for row in state.statistics or []}
    if statistics_tables != set(expected_rows):
        raise RuntimeError(
            f"planner statistics missing for tables: "
            f"{sorted(set(expected_rows) - statistics_tables)}"
        )

    if not state.relations:
        raise RuntimeError("IMDb relation metadata is missing")
    max_frozen_age = max(row.frozen_xid_age for row in state.relations)
    if max_frozen_age > MAX_FRESHLY_FROZEN_XID_AGE:
        raise RuntimeError(
            f"IMDb relations were not freshly frozen: max age={max_frozen_age}"
        )


def representative_query_outputs(
    client: PostgresClient,
    source_dir: Path,
    query_names: list[str],
) -> dict[str, RepresentativeQueryOutput]:
    """Run selected JOB queries and record their exact CSV output, sizes, and hashes."""
    outputs: dict[str, RepresentativeQueryOutput] = {}
    for query_name in query_names:
        query_path = source_dir / query_name
        sql = query_path.read_text(encoding="utf-8")
        output = client.runner_sql(
            sql,
            connection_label="qorl-imdb-query-verifier",
        )
        encoded = output.encode("utf-8")
        outputs[query_name] = RepresentativeQueryOutput(
            bytes=len(encoded), sha256=sha256_bytes(encoded), csv=output
        )
    return outputs


def write_atomic(path: Path, content: str) -> None:
    """Write and flush a temporary file, then atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def verify_load(
    client: PostgresClient,
    output: Path,
    *,
    manifest: ImdbManifest,
    repository: Path = REPOSITORY_ROOT,
) -> None:
    """Verify the loaded database and query results, then write a checksummed report."""
    manifest_path = repository / "scripts/imdb/manifest.json"
    query_dir = repository / "benchmarks/job/queries"

    state = load_database_snapshot(client, manifest.load.table_order)
    validate_database_snapshot(state, manifest)
    query_outputs = representative_query_outputs(
        client=client,
        source_dir=query_dir,
        query_names=[
            JOB_REPRESENTATIVE_QUERY_1,
            JOB_REPRESENTATIVE_QUERY_2,
            JOB_REPRESENTATIVE_QUERY_3,
        ],
    )

    checksums = DatabaseChecksums(
        table_names=section_checksum(state.table_names),
        table_rows=section_checksum(state.table_rows),
        columns=section_checksum(state.columns),
        constraints=section_checksum(state.constraints),
        indexes=section_checksum(state.indexes),
        statistics=section_checksum(state.statistics),
        representative_query_outputs=section_checksum(query_outputs),
    )

    result = LoadVerificationReport(
        fixture_id=manifest.fixture_id,
        captured_at_utc=datetime.now(UTC).isoformat(),
        source_manifest_sha256=sha256_file(manifest_path),
        database=state,
        representative_query_outputs=query_outputs,
        checksums=checksums,
    )

    write_atomic(
        output,
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    print(
        f"imdb load verification passed: "
        f"tables={len(state.table_rows)} rows={sum(state.table_rows.values())} "
        f"indexes={len(state.indexes or [])}"
    )


def archive_database(pool: ContainerPool, slot: WorkerSlot, archive: Path) -> None:
    """Archive a cleanly stopped PostgreSQL volume without overwriting existing files."""
    partial = archive.with_name(f".{archive.name}.part")
    if archive.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite an archive: {archive}")
    state = pool.command(
        [
            "docker",
            "inspect",
            slot.container_id,
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
        pool.command(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--volume",
                f"{slot.volume}:/source:ro",
                "--volume",
                f"{archive.parent}:/output",
                "--entrypoint",
                "bash",
                slot.image_id,
                "-Eeuo",
                "pipefail",
                "-c",
                script,
                "qorl-archive",
                slot.pgdata_relative_path,
            ]
        )
        partial.replace(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    """Load and verify IMDb, archive its stopped volume, and clean up on success."""
    repository = REPOSITORY_ROOT.resolve()
    archive = repository / IMDB_ARCHIVE
    verification_report_path = repository / "data/imdb-verification/loaded.json"
    for path in (
        archive,
        archive.with_name(f".{archive.name}.part"),
        verification_report_path,
    ):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    if not (repository / "data/raw/tables").is_dir():
        raise RuntimeError(
            "IMDb CSVs are missing; run `uv run python -m scripts.imdb.fetch` first"
        )
    manifest = ImdbManifest.model_validate_json(
        (repository / "scripts/imdb/manifest.json").read_text(encoding="utf-8")
    )
    verify_input_csvs_against_manifest(repository, manifest)
    pool_config = load_pool_config(repository, POOL_CONFIG)
    postgres_config = PostgresConfig.load(POSTGRES_CONFIG)
    pool = ContainerPool(repository, "qorl-imdb-load", pool_config, postgres_config)
    if len(pool.workers) != 1:
        raise RuntimeError("IMDb preparation requires a single-worker pool")
    slot = pool.workers[0]
    try:
        pool.create()
        pool.start()
        print("Loading IMDb rows, indexes, and statistics...")
        slot.client.admin_sql(
            (repository / "scripts/imdb/load.sql").read_text(encoding="utf-8")
        )
        pool.command(["docker", "exec", slot.container_id, "qorl-assert-config"])
        pool.load_indexes()
        verify_load(
            client=slot.client,
            output=verification_report_path,
            manifest=manifest,
            repository=repository,
        )
        pool.stop()
        archive_database(pool, slot, archive)
    except BaseException:
        print(f"IMDb load failed; resources retained: {slot.compose_project_name}")
        raise
    pool.close()
    print(f"IMDb loaded, verified, and archived: {archive}")


if __name__ == "__main__":
    main()
