#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
raw_dir="$repository_root/data/raw/job-v1"
project_name="qorl-job-v1-build"
skip_fetch=0

usage() {
    echo "usage: $0 [--raw-dir PATH] [--project-name NAME] [--skip-fetch]" >&2
}

while (($#)); do
    case "$1" in
        --raw-dir)
            raw_dir="$2"
            shift 2
            ;;
        --project-name)
            project_name="$2"
            shift 2
            ;;
        --skip-fetch)
            skip_fetch=1
            shift
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$project_name" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "invalid Compose project name: $project_name" >&2
    exit 2
fi

command -v docker >/dev/null
command -v python3 >/dev/null

if ((skip_fetch == 0)); then
    python3 "$script_dir/fetch-job-v1.py" --raw-dir "$raw_dir"
fi

raw_dir="$(cd -- "$raw_dir" && pwd)"
export QORL_JOB_DATA_DIR="$raw_dir/imdb"
export QORL_JOB_SOURCE_DIR="$raw_dir/source"

compose=(
    docker compose
    --project-name "$project_name"
    --file "$repository_root/compose.yaml"
    --file "$repository_root/compose.job.yaml"
)

"${compose[@]}" config --quiet

volume_name="${project_name}_qorl-postgres-data"
if docker volume inspect "$volume_name" >/dev/null 2>&1; then
    echo "refusing to load into existing Docker volume: $volume_name" >&2
    exit 1
fi

if [[ -n "$("${compose[@]}" ps --all --quiet)" ]]; then
    echo "refusing to reuse existing Compose project: $project_name" >&2
    exit 1
fi

trap 'echo "JOB load failed; Compose project retained for diagnosis: '"$project_name"'" >&2' ERR

"${compose[@]}" up --detach --wait --no-build postgres
container="$("${compose[@]}" ps --quiet postgres)"
if [[ -z "$container" ]]; then
    echo "PostgreSQL container was not created" >&2
    exit 1
fi

public_table_count="$({
    docker exec "$container" bash -Eeuo pipefail -c '
        psql \
            --username="$POSTGRES_USER" \
            --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
            --no-psqlrc \
            --tuples-only \
            --no-align \
            --set=ON_ERROR_STOP=1 \
            --command="SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '\''public'\'' AND c.relkind IN ('\''r'\'', '\''p'\'')"
    '
} | tr -d '[:space:]')"

if [[ "$public_table_count" != "0" ]]; then
    echo "fresh JOB build database unexpectedly contains $public_table_count public tables" >&2
    exit 1
fi

docker exec --interactive "$container" bash -Eeuo pipefail -c '
    psql \
        --username="$POSTGRES_USER" \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --file=-
' < "$script_dir/load-job-v1.sql"

docker exec --interactive "$container" bash -Eeuo pipefail -c '
    psql \
        --username="$POSTGRES_USER" \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --file=-
' < "$script_dir/finalize-job-v1.sql"

docker exec "$container" qorl-assert-benchmark-config

trap - ERR
printf 'job-v1 load and finalization passed: project=%s container=%s volume=%s\n' \
    "$project_name" "$container" "$volume_name"
