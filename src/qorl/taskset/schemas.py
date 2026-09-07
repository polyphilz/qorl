"""Benchmark catalogs, SQL tasks, and experiment task selections."""

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkId(StrEnum):
    JOB = "job"
    CEB = "ceb"


class TaskRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


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


class TemplateMetadata(BaseModel):
    """Query count and audited join topology for one benchmark template."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query_count: int = Field(gt=0)
    topology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkCatalog(BaseModel):
    """SQL task records and the template metadata used to select them."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    benchmark_id: BenchmarkId
    tasks_metadata: dict[str, TemplateMetadata]
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        """Require unique tasks and exact metadata coverage of their templates."""
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task inventory contains duplicate task IDs")
        counts = Counter(task.template_id for task in self.tasks)
        if counts.keys() != self.tasks_metadata.keys():
            raise ValueError("tasks_metadata must contain exactly the task templates")
        prefix = f"{self.benchmark_id.value}-"
        for task in self.tasks:
            if not task.task_id.startswith(prefix) or not task.template_id.startswith(
                prefix
            ):
                raise ValueError(
                    f"task belongs to a different benchmark: {task.task_id}"
                )
        for template_id, count in counts.items():
            if self.tasks_metadata[template_id].query_count != count:
                raise ValueError(f"query_count does not match tasks for {template_id}")
        return self


class TaskSelection(BaseModel):
    """Saved experiment selection; SQL and template metadata stay in the catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    benchmark_id: BenchmarkId
    task_ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        """Reject repeated IDs rather than changing a saved selection."""
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("selection contains duplicate task IDs")
        return self


@dataclass(frozen=True)
class TopologyOwner:
    """The first split and template encountered for a query topology."""

    role: TaskRole
    template_id: str


@dataclass(frozen=True)
class TemplateSample:
    """Requested sample count for a fully qualified template ID."""

    template_id: str
    count: int


@dataclass(frozen=True)
class SelectionExpression:
    """Parsed selection; an empty template tuple selects the whole benchmark."""

    role: TaskRole
    benchmark_id: BenchmarkId
    templates: tuple[TemplateSample, ...] = ()
