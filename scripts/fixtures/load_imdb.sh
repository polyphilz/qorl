#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repository_root/src:$repository_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_profile="$repository_root/docker/worker_pool/configs/000-poolconf-1x32/poolconf.json"
postgres_config="$repository_root/docker/postgres/configs/000-pgconf-default"
raw_dir="$repository_root/imdb/raw"
project_name="qorl-imdb-build"
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
command -v uv >/dev/null
source "$repository_root/scripts/docker/runtime-profile.sh"
qorl_load_postgres_runtime_profile "$repository_root" "$runtime_profile"
source "$repository_root/scripts/docker/postgres-config.sh"
qorl_load_postgres_config "$repository_root" "$postgres_config"

if ((skip_fetch == 0)); then
    uv run --project "$repository_root" --frozen python -m scripts.fixtures.fetch_imdb --raw-dir "$raw_dir"
fi

raw_dir="$(cd -- "$raw_dir" && pwd)"
export QORL_IMDB_DATA_DIR="$raw_dir/tables"
export QORL_IMDB_SOURCE_DIR="$repository_root/benchmarks/raw/job/source"

compose=(
    docker compose
    --project-name "$project_name"
    --file "$repository_root/compose.yaml"
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

trap 'echo "IMDb load failed; Compose project retained for diagnosis: '"$project_name"'" >&2' ERR

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
    echo "fresh IMDb build database unexpectedly contains $public_table_count public tables" >&2
    exit 1
fi

docker exec --interactive "$container" bash -Eeuo pipefail -c '
    psql \
        --username="$POSTGRES_USER" \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --file=-
' < "$script_dir/load_imdb.sql"

docker exec --interactive "$container" bash -Eeuo pipefail -c '
    psql \
        --username="$POSTGRES_USER" \
        --dbname="${POSTGRES_DB:-$POSTGRES_USER}" \
        --no-psqlrc \
        --set=ON_ERROR_STOP=1 \
        --file=-
' < "$script_dir/finalize_imdb.sql"

docker exec "$container" qorl-assert-config

trap - ERR
printf 'imdb load and finalization passed: project=%s container=%s volume=%s\n' \
    "$project_name" "$container" "$volume_name"
