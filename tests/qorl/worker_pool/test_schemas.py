from pathlib import Path

from qorl.postgres.config import PostgresConfig
from qorl.worker_pool.schemas import ComposeEnvironment, PoolConfig


def test_compose_environment_serializes_every_override(
    repository_root: Path, postgres_config: PostgresConfig, pool_config: PoolConfig
) -> None:
    worker = pool_config.workers[0]
    environment = ComposeEnvironment(
        cpuset=worker.cpuset,
        cpuset_mems=worker.cpuset_mems,
        memory_limit=worker.memory_limit,
        memory_bytes=worker.memory_bytes,
        memory_swap_limit=worker.memory_limit,
        memory_swap_bytes=worker.memory_swap_bytes,
        shm_size=worker.shm_size,
        shm_bytes=worker.shm_bytes,
        port=worker.port,
        config_file=postgres_config.pg_conf_path,
        expected_file=postgres_config.expected_path,
        assert_script=repository_root / "docker/postgres/scripts/assert-config.sh",
        dump_script=repository_root / "docker/postgres/scripts/dump-postgres-state.sh",
    )

    assert environment.to_env() == {
        "QORL_POSTGRES_CPUSET": worker.cpuset,
        "QORL_POSTGRES_CPUSET_MEMS": worker.cpuset_mems,
        "QORL_POSTGRES_MEMORY_LIMIT": worker.memory_limit,
        "QORL_POSTGRES_MEMORY_BYTES": str(worker.memory_bytes),
        "QORL_POSTGRES_MEMORY_SWAP_LIMIT": worker.memory_limit,
        "QORL_POSTGRES_MEMORY_SWAP_BYTES": "0",
        "QORL_POSTGRES_SHM_SIZE": worker.shm_size,
        "QORL_POSTGRES_SHM_BYTES": str(worker.shm_bytes),
        "QORL_POSTGRES_PORT": str(worker.port),
        "QORL_POSTGRES_CONFIG_FILE": str(postgres_config.pg_conf_path),
        "QORL_POSTGRES_EXPECTED_FILE": str(postgres_config.expected_path),
        "QORL_POSTGRES_ASSERT_SCRIPT": str(environment.assert_script),
        "QORL_POSTGRES_DUMP_SCRIPT": str(environment.dump_script),
    }
