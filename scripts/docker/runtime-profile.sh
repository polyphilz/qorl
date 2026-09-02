#!/usr/bin/env bash

qorl_load_postgres_runtime_profile() {
    local repository_root="$1"
    local profile="$2"
    local rendered
    local assignment
    local name
    local value

    rendered="$(
        PYTHONPATH="$repository_root/src${PYTHONPATH:+:$PYTHONPATH}" \
        python3 "$repository_root/scripts/docker/profile_env.py" \
            "$profile" --validate-host
    )" || return
    while IFS= read -r assignment; do
        name="${assignment%%=*}"
        value="${assignment#*=}"
        printf -v "$name" '%s' "$value"
        export "${name?}"
    done <<<"$rendered"
}
