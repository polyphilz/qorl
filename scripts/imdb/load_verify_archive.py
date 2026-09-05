"""Load fetched IMDb inputs, verify the database, and archive its stopped volume."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl.db.config import PostgresConfig
from qorl.db.container import PostgresContainer
from qorl.db.fixture import IMDB_ARCHIVE, DatabaseFixture
from qorl.db.resources import load_runtime_profile
from qorl.db.worker import PostgresWorker
from qorl.util.hashing import sha256_bytes, sha256_file
from scripts.imdb.schemas import ImdbManifest
from scripts.shared.verify import verify_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")
POOL_CONFIG = Path("docker/worker_pool/configs/000-poolconf-1x32")
POSTGRES_UID = 999
POSTGRES_GID = 999
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
MAX_FRESHLY_FROZEN_XID_AGE = 1_000


def verify_against_manifest(repository: Path) -> None:
    """Verify extracted IMDb CSV data against the manifest.

    Check file names, byte sizes, and SHA-256 checksums, not individual row values
    or row counts.
    """
    manifest = ImdbManifest.model_validate_json(
        (repository / "scripts/imdb/manifest.json").read_text(encoding="utf-8")
    )
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


def run(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def admin_psql(container: str, sql: str) -> str:
    shell = r"""
exec psql \
    --username="$POSTGRES_USER" \
    --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
    --no-psqlrc \
    --set=ON_ERROR_STOP=1 \
    --quiet \
    --tuples-only \
    --no-align
"""
    return run(
        [
            "docker",
            "exec",
            "--interactive",
            container,
            "bash",
            "-Eeuo",
            "pipefail",
            "-c",
            shell,
        ],
        input_text=sql,
    ).strip()


def runner_psql(container: str, sql: str) -> str:
    shell = r"""
exec env \
    PGPASSWORD="$QORL_RUNNER_PASSWORD" \
    PGAPPNAME=qorl-imdb-query-verifier \
    psql \
        --host=127.0.0.1 \
        --username=qorl_runner \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --quiet \
        --csv
"""
    return run(
        [
            "docker",
            "exec",
            "--interactive",
            container,
            "bash",
            "-Eeuo",
            "pipefail",
            "-c",
            shell,
        ],
        input_text=sql,
    )


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_database_state(container: str, table_names: list[str]) -> dict[str, Any]:
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
    return json.loads(admin_psql(container, sql))


def validate_database_state(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected_rows = {
        member["table"]: member["rows"]
        for member in manifest["dataset"]["members"].values()
        if "table" in member
    }
    if state["table_names"] != sorted(expected_rows):
        raise RuntimeError(
            f"IMDb table-set mismatch: expected={sorted(expected_rows)} "
            f"actual={state['table_names']}"
        )
    if state["table_rows"] != expected_rows:
        mismatches = {
            name: {"expected": expected_rows.get(name), "actual": actual}
            for name, actual in state["table_rows"].items()
            if expected_rows.get(name) != actual
        }
        raise RuntimeError(f"IMDb row-count mismatch: {mismatches}")

    identity = state["identity"]
    database = manifest["database"]
    for key in ("server_version_num", "encoding", "collation", "ctype"):
        if identity[key] != database[key]:
            raise RuntimeError(
                f"database identity mismatch for {key}: "
                f"expected={database[key]} actual={identity[key]}"
            )

    indexes = state["indexes"] or []
    primary_count = sum(index["primary"] for index in indexes)
    secondary_count = sum(not index["primary"] for index in indexes)
    invalid = [
        index["name"] for index in indexes if not index["valid"] or not index["ready"]
    ]
    if len(indexes) != database["expected_total_index_count"]:
        raise RuntimeError(f"unexpected total index count: {len(indexes)}")
    if primary_count != database["expected_primary_key_count"]:
        raise RuntimeError(f"unexpected primary-key index count: {primary_count}")
    if secondary_count != database["expected_secondary_index_count"]:
        raise RuntimeError(f"unexpected secondary index count: {secondary_count}")
    if invalid:
        raise RuntimeError(f"invalid or unready indexes: {invalid}")

    statistics_tables = {row["table"] for row in state["statistics"] or []}
    if statistics_tables != set(expected_rows):
        raise RuntimeError(
            f"planner statistics missing for tables: "
            f"{sorted(set(expected_rows) - statistics_tables)}"
        )

    max_frozen_age = max(row["frozen_xid_age"] for row in state["relations"])
    if max_frozen_age > MAX_FRESHLY_FROZEN_XID_AGE:
        raise RuntimeError(
            f"IMDb relations were not freshly frozen: max age={max_frozen_age}"
        )


def representative_query_outputs(
    container: str, source_dir: Path, query_names: list[str]
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for query_name in query_names:
        query_path = source_dir / query_name
        sql = query_path.read_text(encoding="utf-8")
        output = runner_psql(container, sql)
        encoded = output.encode("utf-8")
        outputs[query_name] = {
            "bytes": len(encoded),
            "sha256": sha256_bytes(encoded),
            "csv": output,
        }
    return outputs


def write_atomic(path: Path, content: str) -> None:
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


def verify(
    container: str,
    output: Path,
    *,
    repository: Path = REPOSITORY_ROOT,
) -> None:
    manifest_path = repository / "scripts/imdb/manifest.json"
    job_manifest = repository / "benchmarks/job/manifest.json"
    query_dir = repository / "benchmarks/job/queries"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = json.loads(job_manifest.read_text(encoding="utf-8"))
    run(
        [
            "docker",
            "exec",
            container,
            "/usr/local/bin/qorl-assert-config",
        ]
    )

    state = load_database_state(container, manifest["load"]["table_order"])
    validate_database_state(state, manifest)
    query_outputs = representative_query_outputs(
        container,
        query_dir,
        job["queries"]["representative"],
    )

    fingerprint_sections = {
        "table_names": state["table_names"],
        "table_rows": state["table_rows"],
        "columns": state["columns"],
        "constraints": state["constraints"],
        "indexes": state["indexes"],
        "statistics": state["statistics"],
        "representative_query_outputs": query_outputs,
    }
    fingerprints = {
        name: fingerprint(value) for name, value in sorted(fingerprint_sections.items())
    }

    result = {
        "schema_version": 1,
        "fixture_id": manifest["fixture_id"],
        "phase": "load",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "database": state,
        "representative_query_outputs": query_outputs,
        "fingerprints": fingerprints,
    }

    write_atomic(output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"imdb load verification passed: "
        f"tables={len(state['table_names'])} rows={sum(state['table_rows'].values())} "
        f"indexes={len(state['indexes'])}"
    )


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


def main() -> None:
    repository = REPOSITORY_ROOT.resolve()
    archive = repository / IMDB_ARCHIVE
    report = repository / "data/imdb-verification/loaded.json"
    for path in (archive, archive.with_name(f".{archive.name}.part"), report):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    if not (repository / "data/raw/tables").is_dir():
        raise RuntimeError(
            "IMDb CSVs are missing; run `uv run python -m scripts.imdb.fetch` first"
        )
    verify_against_manifest(repository)
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


if __name__ == "__main__":
    main()
