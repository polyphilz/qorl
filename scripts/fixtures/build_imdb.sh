#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repository_root/src:$repository_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_profile="$repository_root/docker/worker_pool/configs/000-poolconf-1x32/poolconf.json"
postgres_config="$repository_root/docker/postgres/configs/000-pgconf-default"
raw_dir="$repository_root/imdb/raw"
output_dir="$repository_root/imdb"
build_project="qorl-imdb-build"
restore_project="qorl-imdb-restore"

usage() {
    echo "usage: $0 [--raw-dir PATH] [--output-dir PATH] [--build-project NAME] [--restore-project NAME]" >&2
}

while (($#)); do
    case "$1" in
        --raw-dir)
            raw_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --build-project)
            build_project="$2"
            shift 2
            ;;
        --restore-project)
            restore_project="$2"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ "$build_project" == "$restore_project" ]]; then
    echo "build and restore Compose project names must differ" >&2
    exit 2
fi
source "$repository_root/scripts/docker/runtime-profile.sh"
qorl_load_postgres_runtime_profile "$repository_root" "$runtime_profile"
source "$repository_root/scripts/docker/postgres-config.sh"
qorl_load_postgres_config "$repository_root" "$postgres_config"
for project_name in "$build_project" "$restore_project"; do
    if [[ ! "$project_name" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
        echo "invalid Compose project name: $project_name" >&2
        exit 2
    fi
done

if [[ -e "$output_dir/archive.json" || -e "$output_dir/imdb.tar.gz" || -e "$output_dir/verification" ]]; then
    echo "refusing to overwrite an existing fixture or its verification records: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir/verification"
output_dir="$(cd -- "$output_dir" && pwd)"

trap 'echo "imdb build stopped; failed resources were retained for diagnosis" >&2' ERR

uv run --project "$repository_root" --frozen python -m scripts.fixtures.fetch_imdb --raw-dir "$raw_dir"
raw_dir="$(cd -- "$raw_dir" && pwd)"

source_manifest_copy="$output_dir/build.json"
if [[ "$source_manifest_copy" != "$repository_root/imdb/build.json" ]]; then
    if [[ -e "$source_manifest_copy" ]]; then
        echo "refusing to overwrite build configuration: $source_manifest_copy" >&2
        exit 1
    fi
    cp "$repository_root/imdb/build.json" "$source_manifest_copy"
fi

"$script_dir/load_imdb.sh" \
    --raw-dir "$raw_dir" \
    --project-name "$build_project" \
    --skip-fetch

export QORL_IMDB_DATA_DIR="$raw_dir/tables"
export QORL_IMDB_SOURCE_DIR="$repository_root/benchmarks/raw/job/source"
compose=(
    docker compose
    --project-name "$build_project"
    --file "$repository_root/compose.yaml"
)
container="$("${compose[@]}" ps --quiet postgres)"
build_volume="$(docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}')"
if [[ -z "$build_volume" ]]; then
    echo "could not identify the build PGDATA volume" >&2
    exit 1
fi

uv run --project "$repository_root" --frozen python -m scripts.fixtures.verify_imdb \
    --container "$container" \
    --manifest "$source_manifest_copy" \
    --phase build \
    --output "$output_dir/verification/build.json"

uv run --project "$repository_root" --frozen python -m qorl.db.capture \
    --container "$container" \
    --runtime-profile "$runtime_profile" \
    --postgres-config "$postgres_config" \
    --output-dir "$output_dir/verification" \
    --phase pre

"${compose[@]}" stop --timeout 60 postgres
container="$("${compose[@]}" ps --all --quiet postgres)"

uv run --project "$repository_root" --frozen python -m scripts.fixtures.archive_imdb \
    --container "$container" \
    --manifest "$output_dir/build.json" \
    --build-verification "$output_dir/verification/build.json" \
    --environment-capture "$output_dir/verification/environment.json" \
    --output-dir "$output_dir"

"${compose[@]}" down

"$script_dir/restore_verify_imdb.sh" \
    --archive-manifest "$output_dir/archive.json" \
    --build-verification "$output_dir/verification/build.json" \
    --output-dir "$output_dir/verification" \
    --project-name "$restore_project"

docker volume rm "$build_volume" >/dev/null
trap - ERR

uv run --project "$repository_root" --frozen python - "$output_dir/archive.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    f"imdb build, archive, and clean-room restore passed: "
    f"fixture_id={manifest['fixture_id']} "
    f"archive_sha256={manifest['archive']['sha256']}"
)
PY
