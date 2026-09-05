from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from qorl.postgres.client import PostgresClient

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

    def manifest(self) -> dict[str, int | str]:
        return {
            "slot": self.index,
            "physical_core_count": self.physical_core_count,
            "cpuset": self.cpuset,
            "cpuset_mems": self.cpuset_mems,
            "memory_limit": self.memory_limit,
            "memory_bytes": self.memory_bytes,
            "memory_swap_bytes": self.memory_swap_bytes,
            "shm_size": self.shm_size,
            "shm_bytes": self.shm_bytes,
            "port": self.port,
        }


@dataclass(frozen=True)
class PoolConfig:
    profile_id: str
    path: Path
    sha256: str
    workers: tuple[WorkerResources, ...]
    configuration: WorkerPoolConfig

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "worker_count": len(self.workers),
            "workers": [worker.manifest() for worker in self.workers],
        }


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
