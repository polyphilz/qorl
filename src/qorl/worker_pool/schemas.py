from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field

from qorl.postgres.client import PostgresClient
from qorl.postgres.schemas import PostgresConfigManifest

MAX_TCP_PORT = 65_535


class PoolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PoolWorkerConfig(PoolRecord):
    cpuset: str = Field(min_length=1)
    physical_core_count: int = Field(ge=1)
    port: int = Field(ge=1, le=MAX_TCP_PORT)


class WorkerPoolConfig(PoolRecord):
    memory_limit: str
    shm_size: str
    cpuset_mems: str = Field(min_length=1)
    workers: list[PoolWorkerConfig] = Field(min_length=1)


class ComposeEnvironment(PoolRecord):
    """Per-worker Compose overrides; ordinary process variables are inherited."""

    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            serialization_alias=lambda name: f"QORL_POSTGRES_{name.upper()}"
        )
    )

    cpuset: str
    cpuset_mems: str
    memory_limit: str
    memory_bytes: int
    memory_swap_limit: str
    memory_swap_bytes: int
    shm_size: str
    shm_bytes: int
    port: int
    config_file: Path
    expected_file: Path
    assert_script: Path
    dump_script: Path

    def to_env(self) -> dict[str, str]:
        """Serialize Compose variables to the string values subprocesses require."""
        return {
            name: str(value)
            for name, value in self.model_dump(mode="json", by_alias=True).items()
        }


class WorkerManifest(PoolRecord):
    slot: int
    physical_core_count: int
    cpuset: str
    cpuset_mems: str
    memory_limit: str
    memory_bytes: int
    memory_swap_bytes: int
    shm_size: str
    shm_bytes: int
    port: int


class PoolManifest(PoolRecord):
    id: str
    path: str
    config_sha256: str
    worker_count: int
    workers: list[WorkerManifest]
    postgres_config: PostgresConfigManifest


@dataclass(frozen=True)
class WorkerResources:
    index: int
    physical_core_count: int
    cpuset: str
    cpuset_mems: str
    memory_limit: str
    memory_bytes: int
    memory_swap_bytes: int
    shm_size: str
    shm_bytes: int
    port: int

    def manifest(self) -> WorkerManifest:
        return WorkerManifest(
            slot=self.index,
            physical_core_count=self.physical_core_count,
            cpuset=self.cpuset,
            cpuset_mems=self.cpuset_mems,
            memory_limit=self.memory_limit,
            memory_bytes=self.memory_bytes,
            memory_swap_bytes=self.memory_swap_bytes,
            shm_size=self.shm_size,
            shm_bytes=self.shm_bytes,
            port=self.port,
        )


@dataclass(frozen=True)
class PoolConfig:
    profile_id: str
    path: Path
    sha256: str
    workers: tuple[WorkerResources, ...]
    configuration: WorkerPoolConfig


@dataclass
class WorkerSlot:
    resources: WorkerResources
    compose_project_name: str
    client: PostgresClient = field(init=False)
    container_id: str = ""
    created: bool = False
    volume: str = ""
    image_id: str = ""
    pgdata_relative_path: str = ""
