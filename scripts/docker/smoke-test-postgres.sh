#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"

runtime_profile="$project_root/configs/postgres/evaluation-worker-v1.json"
# shellcheck source=/dev/null
source "$script_dir/runtime-profile.sh"
qorl_load_postgres_runtime_profile "$project_root" "$runtime_profile"

compose=(docker compose --file compose.yaml)

"${compose[@]}" exec --no-TTY postgres qorl-assert-benchmark-config
container="$("${compose[@]}" ps --quiet postgres)"
python3 "$script_dir/check-prompt-settings.py" --container "$container"

# Expansion inside this single-quoted script intentionally happens in the
# container, where the Compose-provided environment variables exist.
# shellcheck disable=SC2016
"${compose[@]}" exec --no-TTY postgres bash -Eeuo pipefail -c '
    database_name="${POSTGRES_DB:-$POSTGRES_USER}"

    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW server_version_num")" = "180006"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW autovacuum")" = "off"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW qorl.benchmark_config_id")" = "benchmark-v2"
    test "$(psql --username="$POSTGRES_USER" --dbname="$database_name" --tuples-only --no-align --command="SHOW geqo")" = "off"
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

    # Exercise the exact relation count at which benchmark-v1 entered GEQO.
    leading_diagnostics="$(psql --username="$POSTGRES_USER" --dbname="$database_name" --set=ON_ERROR_STOP=1 --quiet 2>&1 <<'\''SQL'\''
BEGIN;
SET LOCAL pg_hint_plan.debug_print = detailed;
SET LOCAL pg_hint_plan.message_level = notice;
CREATE TEMP TABLE leading_smoke_01 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_02 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_03 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_04 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_05 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_06 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_07 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_08 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_09 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_10 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_11 (id integer PRIMARY KEY);
CREATE TEMP TABLE leading_smoke_12 (id integer PRIMARY KEY);
EXPLAIN (COSTS OFF)
/*+ Leading(((((((((((s01 s02) s03) s04) s05) s06) s07) s08) s09) s10) s11) s12)) */
SELECT count(*)
FROM leading_smoke_01 AS s01,
     leading_smoke_02 AS s02,
     leading_smoke_03 AS s03,
     leading_smoke_04 AS s04,
     leading_smoke_05 AS s05,
     leading_smoke_06 AS s06,
     leading_smoke_07 AS s07,
     leading_smoke_08 AS s08,
     leading_smoke_09 AS s09,
     leading_smoke_10 AS s10,
     leading_smoke_11 AS s11,
     leading_smoke_12 AS s12
WHERE s01.id = s02.id
  AND s02.id = s03.id
  AND s03.id = s04.id
  AND s04.id = s05.id
  AND s05.id = s06.id
  AND s06.id = s07.id
  AND s07.id = s08.id
  AND s08.id = s09.id
  AND s09.id = s10.id
  AND s10.id = s11.id
  AND s11.id = s12.id;
ROLLBACK;
SQL
)"
    grep --fixed-strings --quiet "{used hints:Leading(" <<<"$leading_diagnostics"

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
