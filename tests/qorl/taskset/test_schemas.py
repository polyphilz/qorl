import pytest
from pydantic import BaseModel

from qorl.taskset.schemas import (
    BenchmarkManifest,
    Relation,
    Task,
)


@pytest.mark.parametrize(
    ("record", "extra"),
    [(BenchmarkManifest, "ignore"), (Relation, "forbid"), (Task, "forbid")],
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
