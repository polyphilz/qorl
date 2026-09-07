import pytest
from pydantic import BaseModel, ValidationError

from qorl.taskset.schemas import (
    BenchmarkCatalog,
    BenchmarkId,
    BenchmarkManifest,
    Relation,
    Task,
    TaskSelection,
    TemplateMetadata,
)
from qorl.taskset.taskset import TaskSet


@pytest.mark.parametrize(
    ("record", "extra"),
    [
        (BenchmarkManifest, "ignore"),
        (Relation, "forbid"),
        (Task, "forbid"),
        (BenchmarkCatalog, "forbid"),
        (TemplateMetadata, "forbid"),
        (TaskSelection, "forbid"),
    ],
)
def test_records_keep_validation_settings(record: type[BaseModel], extra: str) -> None:
    assert record.model_config == {"extra": extra, "frozen": True, "strict": True}


def test_benchmark_manifest_ignores_source_metadata() -> None:
    fields = {
        "schema_version": 3,
        "benchmark_id": "job",
        "fixture_id": "imdb",
        "description": "Join Order Benchmark queries",
    }
    manifest = BenchmarkManifest.model_validate(
        {**fields, "source": {"unmodeled": "metadata"}}
    )

    assert manifest.model_dump() == fields


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("count", "query_count does not match"),
        ("missing_template", "exactly the task templates"),
        ("extra_template", "exactly the task templates"),
        ("duplicate_task", "duplicate task IDs"),
        ("wrong_benchmark", "different benchmark"),
    ],
)
def test_catalog_rejects_inconsistent_metadata(
    benchmark_task_sets: dict[str, TaskSet], change: str, error: str
) -> None:
    source = benchmark_task_sets["job"]
    tasks = list(source.tasks)
    metadata = dict(source.tasks_metadata)
    template_id = tasks[0].template_id
    if change == "count":
        metadata[template_id] = TemplateMetadata(
            query_count=metadata[template_id].query_count + 1,
            topology_sha256=metadata[template_id].topology_sha256,
        )
    elif change == "missing_template":
        del metadata[template_id]
    elif change == "extra_template":
        metadata["job-extra"] = metadata[template_id]
    elif change == "duplicate_task":
        tasks.append(tasks[0])
    with pytest.raises(ValidationError, match=error):
        BenchmarkCatalog(
            benchmark_id=BenchmarkId.CEB
            if change == "wrong_benchmark"
            else BenchmarkId.JOB,
            tasks_metadata=metadata,
            tasks=tasks,
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1"])
def test_template_count_requires_a_positive_integer(count: object) -> None:
    with pytest.raises(ValidationError):
        TemplateMetadata.model_validate(
            {"query_count": count, "topology_sha256": "a" * 64}
        )


@pytest.mark.parametrize("checksum", ["", "a" * 63, "g" * 64, "A" * 64])
def test_template_topology_requires_a_sha256(checksum: str) -> None:
    with pytest.raises(ValidationError, match="topology_sha256"):
        TemplateMetadata(query_count=1, topology_sha256=checksum)


@pytest.mark.parametrize("task_ids", [[], [""], ["job-01a", "job-01a"]])
def test_selection_requires_nonempty_unique_ids(task_ids: list[str]) -> None:
    with pytest.raises(ValidationError):
        TaskSelection(benchmark_id=BenchmarkId.JOB, task_ids=task_ids)
