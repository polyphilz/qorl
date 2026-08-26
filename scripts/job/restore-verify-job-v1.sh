#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
raw_dir="$repository_root/data/raw/job-v1"
project_name="qprl-job-v1-restore"
snapshot_manifest=""
build_verification=""
output_dir=""

usage() {
    echo "usage: $0 --snapshot-manifest PATH --build-verification PATH --output-dir PATH [--raw-dir PATH] [--project-name NAME]" >&2
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

snapshot_manifest="$(cd -- "$(dirname -- "$snapshot_manifest")" && pwd)/$(basename -- "$snapshot_manifest")"
build_verification="$(cd -- "$(dirname -- "$build_verification")" && pwd)/$(basename -- "$build_verification")"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
raw_dir="$(cd -- "$raw_dir" && pwd)"

readarray -t snapshot_fields < <(
    python3 - "$snapshot_manifest" <<'PY'
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
PY
)

expected_archive_sha256="${snapshot_fields[0]}"
snapshot_image_id="${snapshot_fields[1]}"
snapshot_image_reference="${snapshot_fields[2]}"
archive_path="${snapshot_fields[3]}"

actual_archive_sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
if [[ "$actual_archive_sha256" != "$expected_archive_sha256" ]]; then
    echo "snapshot archive checksum mismatch" >&2
    exit 1
fi

actual_image_id="$(docker image inspect "$snapshot_image_reference" --format '{{.Id}}')"
if [[ "$actual_image_id" != "$snapshot_image_id" ]]; then
    echo "snapshot image mismatch: expected=$snapshot_image_id actual=$actual_image_id" >&2
    exit 1
fi

export QPRL_JOB_DATA_DIR="$raw_dir/imdb"
export QPRL_JOB_SOURCE_DIR="$raw_dir/source"
compose=(
    docker compose
    --project-name "$project_name"
    --file "$repository_root/compose.yaml"
    --file "$repository_root/compose.job.yaml"
)

"${compose[@]}" config --quiet

volume_name="${project_name}_qprl-postgres-data"
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
volume_name="$(docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}')"
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
    "$snapshot_image_id" \
    -Eeuo pipefail -c '
        test -z "$(find /target -mindepth 1 -print -quit)"
        gzip --decompress --stdout "/snapshot/'"$archive_name"'" \
            | tar --extract --directory=/target --numeric-owner
        test -f /target/PG_VERSION
        test ! -e /target/postmaster.pid
    '

"${compose[@]}" up --detach --wait --no-build postgres
container="$("${compose[@]}" ps --quiet postgres)"

python3 "$script_dir/verify-job-v1.py" \
    --container "$container" \
    --raw-dir "$raw_dir" \
    --phase restore \
    --compare-to "$build_verification" \
    --output "$output_dir/job-v1.database.restore.json"

python3 "$repository_root/scripts/capture-benchmark-environment.py" \
    --container "$container" \
    --output-dir "$output_dir" \
    --phase post

"${compose[@]}" down --volumes
trap - ERR

printf 'job-v1 clean-room restore verification passed: project=%s\n' "$project_name"
