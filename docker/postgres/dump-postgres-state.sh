#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${QPRL_AGENT_PASSWORD:?QPRL_AGENT_PASSWORD is required}"

database_name="${POSTGRES_DB:-$POSTGRES_USER}"
mode="${1:-}"

admin_psql=(
    psql
    --username "$POSTGRES_USER"
    --dbname "$database_name"
    --no-psqlrc
    --set ON_ERROR_STOP=1
)

case "$mode" in
    identity-json)
        PGAPPNAME=qprl-state-dump "${admin_psql[@]}" \
            --quiet --tuples-only --no-align <<'SQL'
SELECT jsonb_pretty(jsonb_build_object(
    'schema_version', 1,
    'captured_at_utc', to_char(
        clock_timestamp() AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
    ),
    'server_version', current_setting('server_version'),
    'server_version_num', current_setting('server_version_num'),
    'server_build', version(),
    'system_identifier', (
        SELECT system_identifier::text
        FROM pg_control_system()
    ),
    'data_checksums', current_setting('data_checksums'),
    'config_file', current_setting('config_file'),
    'hba_file', current_setting('hba_file'),
    'data_directory', current_setting('data_directory'),
    'jit_available', pg_jit_available(),
    'database', (
        SELECT jsonb_build_object(
            'name', datname,
            'encoding', pg_encoding_to_char(encoding),
            'collation', datcollate,
            'ctype', datctype,
            'locale_provider', datlocprovider
        )
        FROM pg_database
        WHERE datname = current_database()
    ),
    'extensions', (
        SELECT jsonb_object_agg(extname, extversion ORDER BY extname)
        FROM pg_extension
    )
));
SQL
        ;;
    nondefaults-json)
        PGAPPNAME=qprl-state-dump "${admin_psql[@]}" \
            --quiet --tuples-only --no-align <<'SQL'
SELECT jsonb_pretty(jsonb_build_object(
    'schema_version', 1,
    'captured_at_utc', to_char(
        clock_timestamp() AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
    ),
    'definition', 'effective setting differs from PostgreSQL boot_val',
    'settings', COALESCE(
        jsonb_agg(to_jsonb(setting_row) ORDER BY setting_row.name),
        '[]'::jsonb
    )
))
FROM (
    SELECT
        name,
        setting,
        unit,
        boot_val,
        reset_val,
        source,
        sourcefile,
        sourceline,
        pending_restart
    FROM pg_settings
    WHERE setting IS DISTINCT FROM boot_val
    ORDER BY name
) AS setting_row;
SQL
        ;;
    all-settings-csv)
        PGAPPNAME=qprl-state-dump "${admin_psql[@]}" --csv <<'SQL'
SELECT
    name,
    setting,
    unit,
    boot_val,
    reset_val,
    source,
    sourcefile,
    sourceline,
    pending_restart,
    context,
    vartype
FROM pg_settings
ORDER BY name;
SQL
        ;;
    show-all-csv)
        PGPASSWORD="$QPRL_AGENT_PASSWORD" \
        PGAPPNAME=qprl-baseline \
        psql \
            --host 127.0.0.1 \
            --username qp_agent \
            --dbname "$database_name" \
            --no-psqlrc \
            --set ON_ERROR_STOP=1 \
            --csv \
            --command 'SHOW ALL'
        ;;
    *)
        echo "usage: $0 {identity-json|nondefaults-json|all-settings-csv|show-all-csv}" >&2
        exit 2
        ;;
esac
