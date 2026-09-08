from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.model.schemas import JsonValue


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


class Column(BaseModel):
    name: str
    type: str
    nullable: bool


class Index(BaseModel):
    name: str
    definition: str


class ExtendedStatistics(BaseModel):
    name: str
    columns: list[str] | None
    kinds: list[str]


class RelationMetadata(BaseModel):
    exists: bool
    columns: list[Column] = Field(default_factory=list[Column])
    indexes: list[Index] = Field(default_factory=list[Index])
    extended_statistics: list[ExtendedStatistics] = Field(
        default_factory=list[ExtendedStatistics]
    )
    estimated_rows: float | None = None
    table_bytes: int | None = None
    indexes_bytes: int | None = None
    total_bytes: int | None = None
    omitted_columns: int = 0
    omitted_indexes: int = 0
    omitted_extended_statistics: int = 0


class ColumnStatistics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    column: str
    status: Literal["available", "missing_column", "missing_statistics"]
    null_fraction: float | None = None
    average_width_bytes: int | None = None
    n_distinct: float | None = None
    correlation: float | None = None
    most_common_values: list[JsonValue] | None = None
    most_common_frequencies: list[float] | None = None
    histogram_bounds: list[JsonValue] | None = None
    histogram_bound_positions: list[int] = Field(default_factory=list[int])
    common_value_count: int = 0
    histogram_bound_count: int = 0
    omitted_common_values: int = 0
    omitted_histogram_bounds: int = 0
    omitted_common_value_indexes: list[int] = Field(default_factory=list[int])
    omitted_histogram_bound_indexes: list[int] = Field(default_factory=list[int])


class RelationStatistics(BaseModel):
    columns: list[ColumnStatistics]
    histogram_positions_meaning: str = "histogram_bound_positions contains zero-based positions in the original ordered histogram, aligned with histogram_bounds. omitted_histogram_bound_indexes identifies returned-array slots replaced with null for oversized values, not original histogram positions."
    n_distinct_meaning: str = "Positive: estimated distinct count. Negative: fraction of estimated table rows (-1 means unique)."
    correlation_meaning: str = "Correlation of column values with physical heap order, from -1 (reverse) to +1 (same); not correlation between columns."
