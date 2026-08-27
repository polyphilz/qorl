#!/usr/bin/env python3
"""Validate a loaded job-v1 database and emit canonical fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/manifests/job-v1.json"
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data/raw/job-v1"
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def run(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def admin_psql(container: str, sql: str) -> str:
    shell = r'''
exec psql \
    --username="$POSTGRES_USER" \
    --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
    --no-psqlrc \
    --set=ON_ERROR_STOP=1 \
    --quiet \
    --tuples-only \
    --no-align
'''
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
    shell = r'''
exec env \
    PGPASSWORD="$QPRL_RUNNER_PASSWORD" \
    PGAPPNAME=qprl-job-v1-query-verifier \
    psql \
        --host=127.0.0.1 \
        --username=qprl_runner \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --quiet \
        --csv
'''
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

    sql = f"""
WITH expected_tables(table_name) AS (
    SELECT unnest(ARRAY[{table_array}]::text[])
), row_counts AS (
    {row_queries}
), columns AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', c.table_name,
            'ordinal', c.ordinal_position,
            'column', c.column_name,
            'data_type', c.data_type,
            'udt_name', c.udt_name,
            'nullable', c.is_nullable,
            'default', c.column_default,
            'character_maximum_length', c.character_maximum_length,
            'numeric_precision', c.numeric_precision,
            'numeric_scale', c.numeric_scale
        ) ORDER BY c.table_name, c.ordinal_position
    ) AS value
    FROM information_schema.columns AS c
    JOIN expected_tables AS e USING (table_name)
    WHERE c.table_schema = 'public'
), constraints AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'name', constraint_row.conname,
            'type', constraint_row.contype,
            'definition', pg_get_constraintdef(constraint_row.oid, true),
            'validated', constraint_row.convalidated
        ) ORDER BY table_class.relname, constraint_row.conname
    ) AS value
    FROM pg_constraint AS constraint_row
    JOIN pg_class AS table_class ON table_class.oid = constraint_row.conrelid
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
), indexes AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'name', index_class.relname,
            'definition', pg_get_indexdef(index_row.indexrelid),
            'primary', index_row.indisprimary,
            'unique', index_row.indisunique,
            'valid', index_row.indisvalid,
            'ready', index_row.indisready
        ) ORDER BY table_class.relname, index_class.relname
    ) AS value
    FROM pg_index AS index_row
    JOIN pg_class AS table_class ON table_class.oid = index_row.indrelid
    JOIN pg_class AS index_class ON index_class.oid = index_row.indexrelid
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
), statistics AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', stats.tablename,
            'column', stats.attname,
            'inherited', stats.inherited,
            'null_frac', stats.null_frac,
            'avg_width', stats.avg_width,
            'n_distinct', stats.n_distinct,
            'most_common_vals', stats.most_common_vals::text,
            'most_common_freqs', stats.most_common_freqs::text,
            'histogram_bounds', stats.histogram_bounds::text,
            'correlation', stats.correlation,
            'most_common_elems', stats.most_common_elems::text,
            'most_common_elem_freqs', stats.most_common_elem_freqs::text,
            'elem_count_histogram', stats.elem_count_histogram::text
        ) ORDER BY stats.tablename, stats.attname, stats.inherited
    ) AS value
    FROM pg_stats AS stats
    JOIN expected_tables AS e ON e.table_name = stats.tablename
    WHERE stats.schemaname = 'public'
), relations AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'relpages', table_class.relpages,
            'reltuples', table_class.reltuples,
            'relallvisible', table_class.relallvisible,
            'relfrozenxid', table_class.relfrozenxid::text,
            'frozen_xid_age', age(table_class.relfrozenxid),
            'relation_bytes', pg_relation_size(table_class.oid),
            'total_relation_bytes', pg_total_relation_size(table_class.oid)
        ) ORDER BY table_class.relname
    ) AS value
    FROM pg_class AS table_class
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
)
SELECT jsonb_build_object(
    'identity', jsonb_build_object(
        'server_version_num', current_setting('server_version_num'),
        'database', current_database(),
        'encoding', pg_encoding_to_char(database_row.encoding),
        'collation', database_row.datcollate,
        'ctype', database_row.datctype,
        'system_identifier', (SELECT system_identifier::text FROM pg_control_system()),
        'pg_hint_plan_version', (
            SELECT extversion FROM pg_extension WHERE extname = 'pg_hint_plan'
        )
    ),
    'table_names', (
        SELECT jsonb_agg(class_row.relname ORDER BY class_row.relname)
        FROM pg_class AS class_row
        JOIN pg_namespace AS namespace_row ON namespace_row.oid = class_row.relnamespace
        WHERE namespace_row.nspname = 'public'
          AND class_row.relkind IN ('r', 'p')
    ),
    'table_rows', (
        SELECT jsonb_object_agg(table_name, row_count ORDER BY table_name)
        FROM row_counts
    ),
    'columns', (SELECT value FROM columns),
    'constraints', (SELECT value FROM constraints),
    'indexes', (SELECT value FROM indexes),
    'statistics', (SELECT value FROM statistics),
    'relations', (SELECT value FROM relations)
)
FROM pg_database AS database_row
WHERE database_row.datname = current_database();
"""
    return json.loads(admin_psql(container, sql))


def validate_database_state(
    state: dict[str, Any], manifest: dict[str, Any]
) -> None:
    expected_rows = {
        member["table"]: member["rows"]
        for member in manifest["dataset"]["members"].values()
        if "table" in member
    }
    if state["table_names"] != sorted(expected_rows):
        raise RuntimeError(
            f"JOB table-set mismatch: expected={sorted(expected_rows)} "
            f"actual={state['table_names']}"
        )
    if state["table_rows"] != expected_rows:
        mismatches = {
            name: {"expected": expected_rows.get(name), "actual": actual}
            for name, actual in state["table_rows"].items()
            if expected_rows.get(name) != actual
        }
        raise RuntimeError(f"JOB row-count mismatch: {mismatches}")

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
        index["name"]
        for index in indexes
        if not index["valid"] or not index["ready"]
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
    if max_frozen_age > 1000:
        raise RuntimeError(f"JOB relations were not freshly frozen: max age={max_frozen_age}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("build", "restore"))
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run(
        [
            "docker",
            "exec",
            args.container,
            "/usr/local/bin/qprl-assert-benchmark-config",
        ]
    )

    state = load_database_state(
        args.container, manifest["load"]["table_order"]
    )
    validate_database_state(state, manifest)
    query_outputs = representative_query_outputs(
        args.container,
        args.raw_dir / "source",
        manifest["workload"]["queries"]["representative"],
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
        name: fingerprint(value)
        for name, value in sorted(fingerprint_sections.items())
    }

    result = {
        "schema_version": 1,
        "fixture_id": manifest["fixture_id"],
        "phase": args.phase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_file(args.manifest),
        "database": state,
        "representative_query_outputs": query_outputs,
        "fingerprints": fingerprints,
    }

    if args.compare_to:
        expected = json.loads(args.compare_to.read_text(encoding="utf-8"))
        if result["source_manifest_sha256"] != expected["source_manifest_sha256"]:
            raise RuntimeError("source manifest changed between build and restore")
        mismatches = {
            name: {
                "expected": expected["fingerprints"].get(name),
                "actual": actual,
            }
            for name, actual in fingerprints.items()
            if expected["fingerprints"].get(name) != actual
        }
        if mismatches:
            raise RuntimeError(f"restored database fingerprint mismatch: {mismatches}")
        result["compared_to"] = str(args.compare_to)
        result["comparison"] = "passed"

    write_atomic(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"job-v1 {args.phase} verification passed: "
        f"tables={len(state['table_names'])} rows={sum(state['table_rows'].values())} "
        f"indexes={len(state['indexes'])}"
    )


if __name__ == "__main__":
    main()
