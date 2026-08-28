#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"

compose=(docker compose --file compose.yaml)

"${compose[@]}" exec --no-TTY postgres qorl-assert-benchmark-config

# Expansion inside this single-quoted script intentionally happens in the
# container, where the Compose-provided environment variables exist.
# shellcheck disable=SC2016
"${compose[@]}" exec --no-TTY postgres bash -Eeuo pipefail -c '
    database_name="${POSTGRES_DB:-$POSTGRES_USER}"

    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW server_version_num")" = "180006"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW autovacuum")" = "off"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW qorl.benchmark_config_id")" = "benchmark-v1"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW jit")" = "off"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW max_logical_replication_workers")" = "0"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW huge_pages")" = "off"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SELECT extversion FROM pg_extension WHERE extname = '\''pg_hint_plan'\''")" = "1.8.0"
    psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW shared_preload_libraries" | grep --fixed-strings --quiet pg_hint_plan

    hinted_plan="$(psql --username="$POSTGRES_USER" --dbname="$database_name" --set=ON_ERROR_STOP=1 --quiet <<'\''SQL'\''
BEGIN;
CREATE TEMP TABLE hint_smoke (id integer PRIMARY KEY, payload text);
INSERT INTO hint_smoke
SELECT value, repeat('\''x'\'', 32)
FROM generate_series(1, 1000) AS value;
ANALYZE hint_smoke;
EXPLAIN (COSTS OFF)
/*+ IndexScan(hint_smoke hint_smoke_pkey) */
SELECT * FROM hint_smoke WHERE id > 0;
ROLLBACK;
SQL
)"
    grep --extended-regexp --quiet "Index (Only )?Scan using hint_smoke_pkey" <<<"$hinted_plan"

    disabled_plan="$(psql --username="$POSTGRES_USER" --dbname="$database_name" --set=ON_ERROR_STOP=1 --quiet <<'\''SQL'\''
BEGIN;
CREATE TEMP TABLE disable_index_smoke (id integer PRIMARY KEY, payload text);
INSERT INTO disable_index_smoke
SELECT value, repeat('\''x'\'', 32)
FROM generate_series(1, 1000) AS value;
ANALYZE disable_index_smoke;
EXPLAIN (COSTS OFF)
/*+ DisableIndex(disable_index_smoke disable_index_smoke_pkey) */
SELECT * FROM disable_index_smoke WHERE id = 1;
ROLLBACK;
SQL
)"
    if grep --fixed-strings --quiet "disable_index_smoke_pkey" <<<"$disabled_plan"; then
        echo "DisableIndex failed to suppress disable_index_smoke_pkey" >&2
        exit 1
    fi

    test "$(PGPASSWORD="$QORL_RUNNER_PASSWORD" psql --host=127.0.0.1 --username=qorl_runner --dbname="$database_name" --tuples-only --no-align --command="SHOW default_transaction_read_only")" = "on"
    PGPASSWORD="$QORL_RUNNER_PASSWORD" psql --host=127.0.0.1 --username=qorl_runner --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="SELECT 1" >/dev/null

    if PGPASSWORD="$QORL_RUNNER_PASSWORD" psql --host=127.0.0.1 --username=qorl_runner --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="CREATE TABLE forbidden_smoke (id integer)" >/dev/null 2>&1; then
        echo "qorl_runner unexpectedly created a persistent table" >&2
        exit 1
    fi

    if PGPASSWORD="$QORL_RUNNER_PASSWORD" psql --host=127.0.0.1 --username=qorl_runner --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="CREATE TEMP TABLE forbidden_temp_smoke (id integer)" >/dev/null 2>&1; then
        echo "qorl_runner unexpectedly created a temporary table" >&2
        exit 1
    fi
'

echo "QORL PostgreSQL image smoke test passed."
