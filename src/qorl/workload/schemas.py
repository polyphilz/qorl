"""Checked-in benchmark sources and SQL task records."""

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class WorkloadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceArchive(WorkloadRecord):
    url: str
    filename: str
    bytes: int
    sha256: str
    members: int | None = None
    regular_files: int | None = None


class QuerySource(WorkloadRecord):
    repository_url: str
    commit: str
    archive: SourceArchive


class BenchmarkManifest(WorkloadRecord):
    schema_version: int
    workload_id: str
    fixture_id: str = Field(min_length=1)
    description: str
    source: QuerySource


class Relation(WorkloadRecord):
    alias: str
    table: str


class Task(WorkloadRecord):
    task_id: str = Field(min_length=1)
    template_id: str
    sql_path: str
    sql_sha256: str
    tables: list[str]
    relations: list[Relation]
    join_edges: list[str]
    table_count: int
    relation_count: int
    join_predicate_count: int


TASKS_ADAPTER = TypeAdapter(list[Task])
