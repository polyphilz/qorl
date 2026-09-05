#!/usr/bin/env bash

qorl_load_postgres_config() {
    local repository_root="$1"
    local configured="$2"
    local config_dir

    if [[ "$configured" = /* ]]; then
        config_dir="$configured"
    else
        config_dir="$repository_root/$configured"
    fi
    config_dir="$(cd -- "$config_dir" && pwd)" || return
    test -f "$config_dir/pg.conf" || return
    test -f "$config_dir/config.expected.json" || return

    export QORL_POSTGRES_CONFIG_FILE="$config_dir/pg.conf"
    export QORL_POSTGRES_EXPECTED_FILE="$config_dir/config.expected.json"
    export QORL_POSTGRES_ASSERT_SCRIPT="$repository_root/docker/postgres/scripts/assert-config.sh"
    export QORL_POSTGRES_DUMP_SCRIPT="$repository_root/docker/postgres/scripts/dump-postgres-state.sh"
}
