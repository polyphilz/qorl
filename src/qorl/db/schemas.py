from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_TCP_PORT = 65_535


class DatabaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PoolWorkerConfig(DatabaseRecord):
    cpuset: str = Field(min_length=1)
    physical_core_count: int = Field(ge=1)
    port: int = Field(ge=1, le=MAX_TCP_PORT)


class WorkerPoolConfig(DatabaseRecord):
    memory_limit: str
    shm_size: str
    cpuset_mems: str = Field(min_length=1)
    workers: list[PoolWorkerConfig] = Field(min_length=1)


class PostgreSQLExpected(DatabaseRecord):
    server_version_num: str
    extension_name: str
    extension_version: str
    data_checksums: str
    database_encoding: str
    database_collation: str
    database_ctype: str


class PostgresConfigExpected(DatabaseRecord):
    schema_version: int
    postgres_config_id: str
    postgresql: PostgreSQLExpected
    settings: dict[str, str]
    agent_role_settings: dict[str, str]
    forbidden_backend_types: list[str]


class PostgresConfigManifest(DatabaseRecord):
    id: str
    path: str
    pg_conf_sha256: str
    expected_sha256: str


class RuntimeIdentity(DatabaseRecord):
    postgres_image_id: str
    postgres_config_id: str
