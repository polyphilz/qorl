#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${QPRL_RUNNER_PASSWORD:?QPRL_RUNNER_PASSWORD is required on first initialization}"

database_name="${POSTGRES_DB:-$POSTGRES_USER}"

psql \
    --username "$POSTGRES_USER" \
    --dbname "$database_name" \
    --set ON_ERROR_STOP=1 \
    --set bootstrap_user="$POSTGRES_USER" \
    --set database_name="$database_name" \
    --set runner_password="$QPRL_RUNNER_PASSWORD" <<'SQL'
CREATE EXTENSION pg_hint_plan;

REVOKE CREATE, TEMPORARY ON DATABASE :"database_name" FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE ROLE qprl_runner
    LOGIN
    PASSWORD :'runner_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    NOBYPASSRLS;

ALTER ROLE qprl_runner SET default_transaction_read_only = on;
ALTER ROLE qprl_runner SET search_path = public, pg_catalog;

GRANT CONNECT ON DATABASE :"database_name" TO qprl_runner;
GRANT USAGE ON SCHEMA public TO qprl_runner;

-- JOB objects created later by the bootstrap superuser inherit read access.
-- The JOB loader should also issue an explicit GRANT ON ALL TABLES after load
-- so this invariant does not depend only on default privileges.
ALTER DEFAULT PRIVILEGES FOR ROLE :"bootstrap_user" IN SCHEMA public
    GRANT SELECT ON TABLES TO qprl_runner;
SQL
