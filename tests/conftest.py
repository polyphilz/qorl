from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes, PostgresSettings
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.schemas import PoolConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture
def postgres_config() -> PostgresConfig:
    return PostgresConfig.load(Path("docker/postgres/configs/000-pgconf-default"))


@pytest.fixture
def postgres_settings(postgres_config: PostgresConfig) -> PostgresSettings:
    return postgres_config.agent_settings


@pytest.fixture
def postgres_indexes() -> PostgresIndexes:
    return PostgresIndexes(by_table={"title": frozenset({"title_pkey"})})


@pytest.fixture
def pool_config(repository_root: Path) -> PoolConfig:
    return load_pool_config(
        repository_root, Path("docker/worker_pool/configs/002-poolconf-4x8")
    )


@pytest.fixture
def load_experiment(repository_root: Path) -> Callable[[str], dict[str, Any]]:
    def load(relative_path: str) -> dict[str, Any]:
        return runpy.run_path(repository_root / relative_path)

    return load
