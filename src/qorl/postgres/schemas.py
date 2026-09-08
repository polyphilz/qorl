from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ExplainResult:
    document: dict[str, Any]
    hint_diagnostics: str


class DatabaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PostgresIndexes(BaseModel):
    """The pool's public-schema index catalog, populated after database preparation."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    by_table: dict[str, frozenset[str]]


class PlannerSettings(DatabaseRecord):
    """Baseline settings shown to the agent, in PostgreSQL's native string format."""

    enable_bitmapscan: str
    enable_gathermerge: str
    enable_group_by_reordering: str
    enable_hashagg: str
    enable_hashjoin: str
    enable_incremental_sort: str
    enable_indexonlyscan: str
    enable_indexscan: str
    enable_material: str
    enable_memoize: str
    enable_mergejoin: str
    enable_nestloop: str
    enable_parallel_hash: str
    enable_self_join_elimination: str
    enable_seqscan: str
    enable_sort: str
    seq_page_cost: str
    random_page_cost: str
    cpu_tuple_cost: str
    cpu_index_tuple_cost: str
    cpu_operator_cost: str
    parallel_setup_cost: str
    parallel_tuple_cost: str
    effective_cache_size: str


class PostgresResourceLimits(DatabaseRecord):
    work_mem: str
    max_worker_processes: str
    max_parallel_workers: str
    max_parallel_workers_per_gather: str
    parallel_leader_participation: str


class PostgresSettings(PlannerSettings, PostgresResourceLimits):
    """Verified planner settings and resource limits in native PostgreSQL units."""


class WorkerAllocation(DatabaseRecord):
    """Container limits verified at worker startup, not host-wide resources."""

    cpuset: str
    physical_core_count: int
    memory_bytes: int


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
