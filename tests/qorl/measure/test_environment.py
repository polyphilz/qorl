import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl.measure.environment import capture_environment
from qorl.postgres.config import PostgresConfig
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_capture_uses_the_active_worker_and_selected_configs(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
    phase: str,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
) -> None:
    pool = ContainerPool(repository_root, "capture-test", pool_config, postgres_config)
    slot = pool.workers[0]
    slot.container_id = "active-container"
    command = Mock()
    monkeypatch.setattr(pool, "command", command)

    capture_environment(pool, slot, tmp_path, phase)

    command.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "qorl.measure.environment",
            "--container",
            "active-container",
            "--output-dir",
            str(tmp_path),
            "--phase",
            phase,
            "--runtime-profile",
            str(repository_root / pool_config.path),
            "--postgres-config",
            str(pool.postgres_config.path),
        ]
    )
