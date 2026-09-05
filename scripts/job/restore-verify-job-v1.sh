#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repository_root/src:$repository_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_profile="$repository_root/docker/worker_pool/configs/000-poolconf-1x32/poolconf.json"
postgres_config="$repository_root/docker/postgres/configs/000-pgconf-default"
raw_dir="$repository_root/benchmarks/raw/job-v1"
project_name="qorl-job-v1-restore"
snapshot_manifest=""
build_verification=""
output_dir=""
refresh_runtime_identity=0

usage() {
    echo "usage: $0 --snapshot-manifest PATH --build-verification PATH --output-dir PATH [--raw-dir PATH] [--project-name NAME] [--refresh-runtime-identity]" >&2
}

while (($#)); do
    case "$1" in
        --snapshot-manifest)
            snapshot_manifest="$2"
            shift 2
            ;;
        --build-verification)
            build_verification="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --raw-dir)
            raw_dir="$2"
            shift 2
            ;;
        --project-name)
            project_name="$2"
            shift 2
            ;;
        --refresh-runtime-identity)
            refresh_runtime_identity=1
            shift
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$snapshot_manifest" || -z "$build_verification" || -z "$output_dir" ]]; then
    usage
    exit 2
fi
if [[ ! "$project_name" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "invalid Compose project name: $project_name" >&2
    exit 2
fi
source "$repository_root/scripts/docker/runtime-profile.sh"
qorl_load_postgres_runtime_profile "$repository_root" "$runtime_profile"
source "$repository_root/scripts/docker/postgres-config.sh"
qorl_load_postgres_config "$repository_root" "$postgres_config"

snapshot_manifest="$(cd -- "$(dirname -- "$snapshot_manifest")" && pwd)/$(basename -- "$snapshot_manifest")"
build_verification="$(cd -- "$(dirname -- "$build_verification")" && pwd)/$(basename -- "$build_verification")"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
raw_dir="$(cd -- "$raw_dir" && pwd)"

readarray -t snapshot_fields < <(
    uv run --project "$repository_root" --frozen python - "$snapshot_manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
archive_name = manifest["archive"]["filename"]
if Path(archive_name).name != archive_name:
    raise SystemExit("snapshot archive filename is not a basename")
print(manifest["archive"]["sha256"])
print(manifest["image"]["id"])
print(manifest["image"]["reference"])
print(str(manifest_path.parent / archive_name))
relative = Path(manifest["postgresql"]["pgdata_volume_relative_path"])
if relative.is_absolute() or ".." in relative.parts:
    raise SystemExit("invalid PGDATA path in snapshot manifest")
print(relative.as_posix())
PY
)

expected_archive_sha256="${snapshot_fields[0]}"
snapshot_image_id="${snapshot_fields[1]}"
snapshot_image_reference="${snapshot_fields[2]}"
archive_path="${snapshot_fields[3]}"
pgdata_relative_path="${snapshot_fields[4]}"

actual_archive_sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
if [[ "$actual_archive_sha256" != "$expected_archive_sha256" ]]; then
    echo "snapshot archive checksum mismatch" >&2
    exit 1
fi

actual_image_id="$(docker image inspect "$snapshot_image_reference" --format '{{.Id}}')"
if [[ "$actual_image_id" != "$snapshot_image_id" && "$refresh_runtime_identity" -eq 0 ]]; then
    echo "snapshot image mismatch: expected=$snapshot_image_id actual=$actual_image_id" >&2
    exit 1
fi
restore_image_id="$snapshot_image_id"
if [[ "$refresh_runtime_identity" -eq 1 ]]; then
    restore_image_id="$actual_image_id"
fi

export QORL_JOB_DATA_DIR="$raw_dir/imdb"
export QORL_JOB_SOURCE_DIR="$raw_dir/source"
compose=(
    docker compose
    --project-name "$project_name"
    --file "$repository_root/compose.yaml"
    --file "$repository_root/compose.fixture-build.yaml"
)

"${compose[@]}" config --quiet

volume_name="${project_name}_qorl-postgres-data"
if docker volume inspect "$volume_name" >/dev/null 2>&1; then
    echo "refusing to restore into existing Docker volume: $volume_name" >&2
    exit 1
fi
if [[ -n "$("${compose[@]}" ps --all --quiet)" ]]; then
    echo "refusing to reuse existing Compose project: $project_name" >&2
    exit 1
fi

trap 'echo "JOB restore verification failed; Compose project retained for diagnosis: '"$project_name"'" >&2' ERR

"${compose[@]}" create --no-build postgres
container="$("${compose[@]}" ps --all --quiet postgres)"
volume_name="$(docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}')"
if [[ -z "$volume_name" ]]; then
    echo "could not identify the restore PGDATA volume" >&2
    exit 1
fi

archive_dir="$(dirname -- "$archive_path")"
archive_name="$(basename -- "$archive_path")"
docker run \
    --rm \
    --network=none \
    --volume "$volume_name:/target" \
    --volume "$archive_dir:/snapshot:ro" \
    --entrypoint bash \
    "$restore_image_id" \
    -Eeuo pipefail -c '
        test -z "$(find /target -mindepth 1 -print -quit)"
        mkdir -p "/target/$1"
        gzip --decompress --stdout "/snapshot/'"$archive_name"'" \
            | tar --extract --directory="/target/$1" --numeric-owner
        test -f "/target/$1/PG_VERSION"
        test ! -e "/target/$1/postmaster.pid"
    ' qorl-restore "$pgdata_relative_path"

"${compose[@]}" up --detach --wait --no-build postgres
container="$("${compose[@]}" ps --quiet postgres)"

uv run --project "$repository_root" --frozen python -m scripts.job.verify_job_v1 \
    --container "$container" \
    --raw-dir "$raw_dir" \
    --phase restore \
    --compare-to "$build_verification" \
    --output "$output_dir/job-v1.database.restore.json"

uv run --project "$repository_root" --frozen python -m qorl.db.capture \
    --container "$container" \
    --runtime-profile "$runtime_profile" \
    --postgres-config "$postgres_config" \
    --output-dir "$output_dir" \
    --phase post

"${compose[@]}" down --volumes
trap - ERR

if [[ "$refresh_runtime_identity" -eq 1 ]]; then
    uv run --project "$repository_root" --frozen python "$script_dir/update_snapshot_runtime.py" \
        --snapshot-manifest "$snapshot_manifest"
fi

printf 'job-v1 clean-room restore verification passed: project=%s\n' "$project_name"
