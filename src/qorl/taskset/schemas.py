"""Benchmark metadata used by the loader and SQL task records."""

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_version: int
    benchmark_id: str
    fixture_id: str = Field(min_length=1)
    description: str


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    alias: str
    table: str


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

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
