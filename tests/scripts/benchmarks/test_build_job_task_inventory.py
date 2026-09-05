from pathlib import Path

import pytest

from scripts.benchmarks.build_job_task_inventory import build_inventory, write_inventory


def test_build_inventory_has_no_experiment_roles(repository_root: Path) -> None:
    inventory, queries = build_inventory(
        repository_root / "benchmarks/job/manifest.json",
        repository_root / "benchmarks/job/queries",
    )

    assert inventory["schema_version"] == 3
    assert inventory["fixture_id"] == "imdb"
    assert "database" not in inventory
    assert inventory["task_count"] == len(queries) == 113
    assert (
        not {"role", "split", "training_allowed", "tuning_allowed"} & inventory.keys()
    )
    assert all("partition" not in task for task in inventory["tasks"])


def test_build_preserves_source_manifest(tmp_path: Path) -> None:
    output = tmp_path / "job"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text('{"workload_id": "job"}\n')
    query = tmp_path / "1a.sql"
    query.write_text("SELECT 1;\n")

    write_inventory(output, {"tasks": []}, [query])

    assert manifest.read_text() == '{"workload_id": "job"}\n'
    assert (output / "queries/1a.sql").read_bytes() == query.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_inventory(output, {"tasks": []}, [query])
