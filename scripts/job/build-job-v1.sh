#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repository_root/src:$repository_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_profile="$repository_root/docker/worker_pool/configs/000-poolconf-1x32/poolconf.json"
postgres_config="$repository_root/docker/postgres/configs/000-pgconf-default"
raw_dir="$repository_root/benchmarks/raw/job-v1"
output_dir="$repository_root/artifacts/job-v1"
build_project="qorl-job-v1-build"
restore_project="qorl-job-v1-restore"

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

if [[ -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to overwrite non-empty output directory: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

trap 'echo "job-v1 build stopped; failed resources were retained for diagnosis" >&2' ERR

uv run --project "$repository_root" --frozen python -m scripts.job.fetch_job_v1 --raw-dir "$raw_dir"
raw_dir="$(cd -- "$raw_dir" && pwd)"

source_manifest_copy="$output_dir/job-v1.source.json"
if [[ -e "$source_manifest_copy" ]]; then
    echo "refusing to overwrite source manifest copy: $source_manifest_copy" >&2
    exit 1
fi
cp "$repository_root/benchmarks/job/manifest.json" "$source_manifest_copy"

"$script_dir/load-job-v1.sh" \
    --raw-dir "$raw_dir" \
    --project-name "$build_project" \
    --skip-fetch

export QORL_JOB_DATA_DIR="$raw_dir/imdb"
export QORL_JOB_SOURCE_DIR="$raw_dir/source"
compose=(
    docker compose
    --project-name "$build_project"
    --file "$repository_root/compose.yaml"
    --file "$repository_root/compose.fixture-build.yaml"
)
container="$("${compose[@]}" ps --quiet postgres)"
build_volume="$(docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}')"
if [[ -z "$build_volume" ]]; then
    echo "could not identify the build PGDATA volume" >&2
    exit 1
fi

uv run --project "$repository_root" --frozen python -m scripts.job.verify_job_v1 \
    --container "$container" \
    --raw-dir "$raw_dir" \
    --phase build \
    --output "$output_dir/job-v1.database.build.json"

uv run --project "$repository_root" --frozen python -m qorl.db.capture \
    --container "$container" \
    --runtime-profile "$runtime_profile" \
    --postgres-config "$postgres_config" \
    --output-dir "$output_dir" \
    --phase pre

"${compose[@]}" stop --timeout 60 postgres
container="$("${compose[@]}" ps --all --quiet postgres)"

uv run --project "$repository_root" --frozen python -m scripts.job.seal_job_v1 \
    --container "$container" \
    --manifest "$output_dir/job-v1.source.json" \
    --build-verification "$output_dir/job-v1.database.build.json" \
    --environment-capture "$output_dir/environment.json" \
    --output-dir "$output_dir"

"${compose[@]}" down

"$script_dir/restore-verify-job-v1.sh" \
    --snapshot-manifest "$output_dir/job-v1.snapshot.json" \
    --build-verification "$output_dir/job-v1.database.build.json" \
    --output-dir "$output_dir" \
    --raw-dir "$raw_dir" \
    --project-name "$restore_project"

docker volume rm "$build_volume" >/dev/null
trap - ERR

uv run --project "$repository_root" --frozen python - "$output_dir/job-v1.snapshot.json" <<'PY'
import json
import sys
from pathlib import Path

snapshot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    f"job-v1 build, seal, and clean-room restore passed: "
    f"snapshot_id={snapshot['snapshot_id']} "
    f"archive_sha256={snapshot['archive']['sha256']}"
)
PY
