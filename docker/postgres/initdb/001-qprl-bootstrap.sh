#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${QPRL_AGENT_PASSWORD:?QPRL_AGENT_PASSWORD is required on first initialization}"

database_name="${POSTGRES_DB:-$POSTGRES_USER}"

psql \
    --username "$POSTGRES_USER" \
    --dbname "$database_name" \
    --set ON_ERROR_STOP=1 \
    --set bootstrap_user="$POSTGRES_USER" \
    --set database_name="$database_name" \
    --set agent_password="$QPRL_AGENT_PASSWORD" <<'SQL'
CREATE EXTENSION pg_hint_plan;

REVOKE CREATE, TEMPORARY ON DATABASE :"database_name" FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE ROLE qp_agent
    LOGIN
    PASSWORD :'agent_password'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    NOBYPASSRLS;

ALTER ROLE qp_agent SET default_transaction_read_only = on;
ALTER ROLE qp_agent SET search_path = public, pg_catalog;

GRANT CONNECT ON DATABASE :"database_name" TO qp_agent;
GRANT USAGE ON SCHEMA public TO qp_agent;

-- JOB objects created later by the bootstrap superuser inherit read access.
-- The JOB loader should also issue an explicit GRANT ON ALL TABLES after load
-- so this invariant does not depend only on default privileges.
ALTER DEFAULT PRIVILEGES FOR ROLE :"bootstrap_user" IN SCHEMA public
    GRANT SELECT ON TABLES TO qp_agent;
SQL
