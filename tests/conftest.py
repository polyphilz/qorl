from __future__ import annotations

import runpy
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qorl.measure.schemas import RolloutRecord
from qorl.model.schemas import ModelPreset
from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes, PostgresSettings
from qorl.rl.schemas import RlRolloutRecord
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture
def openrouter_preset(repository_root: Path) -> ModelPreset:
    return ModelPreset.model_validate(
        tomllib.loads(
            (
                repository_root / "configs/defaults/models/000-qwen3.8-2.4t-a95b.toml"
            ).read_text()
        )
    )


@pytest.fixture(scope="session")
def benchmark_task_sets(repository_root: Path) -> dict[str, TaskSet]:
    return {
        benchmark: TaskSet.load(repository_root, benchmark)
        for benchmark in ("job", "ceb")
    }


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
def pool_config() -> PoolConfig:
    return load_pool_config(Path("docker/worker_pool/configs/002-poolconf-4x8"))


@pytest.fixture
def rollout_record(repository_root: Path) -> RolloutRecord:
    return RolloutRecord.model_validate_json(
        (repository_root / "tests/qorl/measure/golden_rollout.json").read_text()
    )


@pytest.fixture
def rl_rollout_record(
    rollout_record: RolloutRecord,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
) -> RlRolloutRecord:
    pool = ContainerPool("test-record", pool_config, postgres_config)
    return RlRolloutRecord(
        **rollout_record.model_dump(),
        database_pool=pool.manifest(),
        database_worker=pool_config.workers[0].manifest(),
        scalar_reward=0.15,
    )


@pytest.fixture
def load_experiment(repository_root: Path) -> Callable[[str], dict[str, Any]]:
    def load(relative_path: str) -> dict[str, Any]:
        return runpy.run_path(repository_root / relative_path)

    return load
