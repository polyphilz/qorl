#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"

compose=(docker compose --file compose.yaml)

# Expansion inside this single-quoted script intentionally happens in the
# container, where the Compose-provided environment variables exist.
# shellcheck disable=SC2016
"${compose[@]}" exec --no-TTY postgres bash -Eeuo pipefail -c '
    database_name="${POSTGRES_DB:-$POSTGRES_USER}"

    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW server_version_num")" = "160015"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW autovacuum")" = "off"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SELECT extversion FROM pg_extension WHERE extname = '\''pg_hint_plan'\''")" = "1.6.2"
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

    test "$(PGPASSWORD="$QPRL_AGENT_PASSWORD" psql --host=127.0.0.1 --username=qp_agent --dbname="$database_name" --tuples-only --no-align --command="SHOW default_transaction_read_only")" = "on"
    PGPASSWORD="$QPRL_AGENT_PASSWORD" psql --host=127.0.0.1 --username=qp_agent --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="SELECT 1" >/dev/null

    if PGPASSWORD="$QPRL_AGENT_PASSWORD" psql --host=127.0.0.1 --username=qp_agent --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="CREATE TABLE forbidden_smoke (id integer)" >/dev/null 2>&1; then
        echo "qp_agent unexpectedly created a persistent table" >&2
        exit 1
    fi

    if PGPASSWORD="$QPRL_AGENT_PASSWORD" psql --host=127.0.0.1 --username=qp_agent --dbname="$database_name" --set=ON_ERROR_STOP=1 --command="CREATE TEMP TABLE forbidden_temp_smoke (id integer)" >/dev/null 2>&1; then
        echo "qp_agent unexpectedly created a temporary table" >&2
        exit 1
    fi
'

echo "QPRL PostgreSQL image smoke test passed."
