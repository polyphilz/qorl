#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 0)); then
    echo "usage: $0" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
postgres_dir="$(cd -- "$script_dir/.." && pwd)"
configs_dir="$postgres_dir/configs"
default_id="000-pgconf-default"
default_dir="$configs_dir/$default_id"

for filename in pg.conf config.expected.json; do
    if [[ ! -f "$default_dir/$filename" ]]; then
        echo "default PostgreSQL config is missing $filename" >&2
        exit 1
    fi
done

highest=-1
shopt -s nullglob
for config_dir in "$configs_dir"/[0-9][0-9][0-9]-pgconf*; do
    config_name="$(basename -- "$config_dir")"
    prefix="${config_name%%-*}"
    number=$((10#$prefix))
    if ((number > highest)); then
        highest=$number
    fi
done

next=$((highest + 1))
if ((next > 999)); then
    echo "PostgreSQL config numbering has exceeded three digits" >&2
    exit 1
fi
new_id="$(printf '%03d-pgconf' "$next")"
target="$configs_dir/$new_id"
if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing PostgreSQL config: $target" >&2
    exit 1
fi

staging="$(mktemp -d "$configs_dir/.new-${new_id}.XXXXXX")"
cleanup() {
    rm -rf -- "$staging"
}
trap cleanup EXIT

sed "s/$default_id/$new_id/g" "$default_dir/pg.conf" >"$staging/pg.conf"
sed "s/$default_id/$new_id/g" \
    "$default_dir/config.expected.json" \
    >"$staging/config.expected.json"
printf '# %s\n\nFill me out!\n' "$new_id" >"$staging/README.md"

mv -- "$staging" "$target"
trap - EXIT
printf 'Created %s\n' "$target"
