#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${QORL_RUNNER_PASSWORD:?QORL_RUNNER_PASSWORD is required}"

database_name="${POSTGRES_DB:-$POSTGRES_USER}"

admin_psql=(
    psql
    --username "$POSTGRES_USER"
    --dbname "$database_name"
    --no-psqlrc
    --set ON_ERROR_STOP=1
    --quiet
)

PGAPPNAME=qorl-config-assert "${admin_psql[@]}" <<'SQL'
DO $assert$
DECLARE
    contract jsonb := pg_read_file('/usr/share/qorl/benchmark-v2.expected.json')::jsonb;
    mismatch text;
    actual_value text;
BEGIN
    WITH expected AS (
        SELECT key AS name, value AS expected_setting
        FROM jsonb_each_text(contract -> 'settings')
    ), mismatches AS (
        SELECT
            expected.name,
            expected.expected_setting,
            actual.setting AS actual_setting
        FROM expected
        LEFT JOIN pg_settings AS actual USING (name)
        WHERE actual.name IS NULL
           OR actual.setting IS DISTINCT FROM expected.expected_setting
    )
    SELECT string_agg(
        format('%s expected=%L actual=%L', name, expected_setting, actual_setting),
        E'\n' ORDER BY name
    )
    INTO mismatch
    FROM mismatches;

    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'benchmark-v2 PostgreSQL setting mismatch:%', E'\n' || mismatch;
    END IF;

    SELECT string_agg(name, ', ' ORDER BY name)
    INTO mismatch
    FROM pg_settings
    WHERE pending_restart;

    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'settings pending restart: %', mismatch;
    END IF;

    SELECT string_agg(format('%s:%s: %s', sourcefile, sourceline, error), E'\n')
    INTO mismatch
    FROM pg_file_settings
    WHERE error IS NOT NULL;

    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'PostgreSQL configuration-file errors:%', E'\n' || mismatch;
    END IF;

    actual_value := current_setting('qorl.benchmark_config_id', true);
    IF actual_value IS DISTINCT FROM contract ->> 'benchmark_config_id' THEN
        RAISE EXCEPTION 'benchmark config ID expected=% actual=%',
            contract ->> 'benchmark_config_id', actual_value;
    END IF;

    actual_value := current_setting('server_version_num');
    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,server_version_num}' THEN
        RAISE EXCEPTION 'server_version_num expected=% actual=%',
            contract #>> '{postgresql,server_version_num}', actual_value;
    END IF;

    actual_value := current_setting('data_checksums');
    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,data_checksums}' THEN
        RAISE EXCEPTION 'data_checksums expected=% actual=%',
            contract #>> '{postgresql,data_checksums}', actual_value;
    END IF;

    SELECT extversion
    INTO actual_value
    FROM pg_extension
    WHERE extname = contract #>> '{postgresql,extension_name}';

    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,extension_version}' THEN
        RAISE EXCEPTION 'extension version expected=% actual=%',
            contract #>> '{postgresql,extension_version}', actual_value;
    END IF;

    SELECT pg_encoding_to_char(encoding)
    INTO actual_value
    FROM pg_database
    WHERE datname = current_database();

    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,database_encoding}' THEN
        RAISE EXCEPTION 'database encoding expected=% actual=%',
            contract #>> '{postgresql,database_encoding}', actual_value;
    END IF;

    SELECT datcollate
    INTO actual_value
    FROM pg_database
    WHERE datname = current_database();

    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,database_collation}' THEN
        RAISE EXCEPTION 'database collation expected=% actual=%',
            contract #>> '{postgresql,database_collation}', actual_value;
    END IF;

    SELECT datctype
    INTO actual_value
    FROM pg_database
    WHERE datname = current_database();

    IF actual_value IS DISTINCT FROM contract #>> '{postgresql,database_ctype}' THEN
        RAISE EXCEPTION 'database ctype expected=% actual=%',
            contract #>> '{postgresql,database_ctype}', actual_value;
    END IF;

    SELECT string_agg(activity.backend_type, ', ' ORDER BY activity.backend_type)
    INTO mismatch
    FROM pg_stat_activity AS activity
    JOIN LATERAL jsonb_array_elements_text(contract -> 'forbidden_backend_types') AS forbidden(backend_type)
      ON forbidden.backend_type = activity.backend_type;

    IF mismatch IS NOT NULL THEN
        RAISE EXCEPTION 'forbidden benchmark background processes are running: %', mismatch;
    END IF;
END
$assert$;
SQL

PGPASSWORD="$QORL_RUNNER_PASSWORD" \
PGAPPNAME=qorl-runner-config-assert \
psql \
    --host 127.0.0.1 \
    --username qorl_runner \
    --dbname "$database_name" \
    --no-psqlrc \
    --set ON_ERROR_STOP=1 \
    --quiet <<'SQL'
DO $assert$
BEGIN
    IF current_user IS DISTINCT FROM 'qorl_runner' THEN
        RAISE EXCEPTION 'expected qorl_runner login, actual=%', current_user;
    END IF;

    IF current_setting('default_transaction_read_only') IS DISTINCT FROM 'on' THEN
        RAISE EXCEPTION 'qorl_runner default_transaction_read_only is not on';
    END IF;

    IF current_setting('transaction_read_only') IS DISTINCT FROM 'on' THEN
        RAISE EXCEPTION 'qorl_runner transaction_read_only is not on';
    END IF;

    IF current_setting('search_path') IS DISTINCT FROM 'public, pg_catalog' THEN
        RAISE EXCEPTION 'qorl_runner search_path expected=% actual=%',
            'public, pg_catalog', current_setting('search_path');
    END IF;
END
$assert$;
SQL

assert_equal() {
    local label="$1"
    local expected="$2"
    local actual="$3"

    if [[ "$actual" != "$expected" ]]; then
        printf '%s expected=%q actual=%q\n' "$label" "$expected" "$actual" >&2
        return 1
    fi
}

if [[ "${QORL_ASSERT_RUNTIME:-0}" == "1" ]]; then
    : "${QORL_EXPECTED_CPUSET:?QORL_EXPECTED_CPUSET is required}"
    : "${QORL_EXPECTED_CPUSET_MEMS:?QORL_EXPECTED_CPUSET_MEMS is required}"
    : "${QORL_EXPECTED_MEMORY_BYTES:?QORL_EXPECTED_MEMORY_BYTES is required}"
    : "${QORL_EXPECTED_MEMORY_SWAP_BYTES:?QORL_EXPECTED_MEMORY_SWAP_BYTES is required}"
    : "${QORL_EXPECTED_SHM_BYTES:?QORL_EXPECTED_SHM_BYTES is required}"

    cpus_allowed="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
    mems_allowed="$(awk '/^Mems_allowed_list:/ {print $2}' /proc/self/status)"
    assert_equal "Cpus_allowed_list" "$QORL_EXPECTED_CPUSET" "$cpus_allowed"
    assert_equal "Mems_allowed_list" "$QORL_EXPECTED_CPUSET_MEMS" "$mems_allowed"

    test -r /sys/fs/cgroup/memory.max
    test -r /sys/fs/cgroup/memory.swap.max
    assert_equal \
        "cgroup memory.max" \
        "$QORL_EXPECTED_MEMORY_BYTES" \
        "$(</sys/fs/cgroup/memory.max)"
    assert_equal \
        "cgroup memory.swap.max" \
        "$QORL_EXPECTED_MEMORY_SWAP_BYTES" \
        "$(</sys/fs/cgroup/memory.swap.max)"

    read -r shm_block_size shm_blocks \
        < <(stat --file-system --format='%S %b' /dev/shm)
    shm_size_bytes=$((shm_block_size * shm_blocks))
    assert_equal "shared memory bytes" "$QORL_EXPECTED_SHM_BYTES" "$shm_size_bytes"
fi

echo "QORL benchmark-v2 configuration assertions passed."
