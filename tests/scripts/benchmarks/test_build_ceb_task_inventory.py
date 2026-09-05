import json
from pathlib import Path

import pytest

from scripts.benchmarks.build_ceb_task_inventory import build_inventory


@pytest.mark.parametrize("same_sql_across_templates", [False, True])
def test_build_neutral_catalog_deduplicates_only_within_templates(
    repository_root: Path, tmp_path: Path, same_sql_across_templates: bool
) -> None:
    ceb = tmp_path / "ceb"
    provenance = ceb / "provenance"
    provenance.mkdir(parents=True)
    original = repository_root / "benchmarks/ceb"
    (ceb / "manifest.json").write_bytes((original / "manifest.json").read_bytes())
    sources = json.loads((original / "provenance/sources.json").read_text())
    templates = sorted(sources["template_query_counts"])[:2]
    selected = [
        next(item for item in sources["queries"] if item["template_id"] == template)
        for template in templates
    ]
    if same_sql_across_templates:
        selected[1]["sql_sha256"] = selected[0]["sql_sha256"]
    duplicate = {
        **selected[0],
        "source_id": selected[0]["source_id"] + "-copy",
        "task_id": selected[0]["task_id"] + "-copy",
    }
    sources["queries"] = [*selected, duplicate]
    sources["template_query_counts"] = {templates[0]: 2, templates[1]: 1}
    (provenance / "sources.json").write_text(json.dumps(sources))
    (provenance / "unique-plans.json").write_text('{"members": []}\n')
    for source in selected:
        target = ceb / source["sql_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        original_source = selected[0] if same_sql_across_templates else source
        target.write_bytes((original / original_source["sql_path"]).read_bytes())
    inventory = build_inventory(ceb)

    assert inventory["schema_version"] == 4
    assert inventory["task_count"] == inventory["template_count"] == 2
    assert {task["task_id"] for task in inventory["tasks"]} == {
        source["task_id"] for source in selected
    }
    assert (
        not {"role", "split", "training_allowed", "tuning_allowed"} & inventory.keys()
    )
    assert all("partition" not in task for task in inventory["tasks"])
    assert inventory["tasks"][0]["duplicate_source_ids"] == [duplicate["source_id"]]
    assert inventory["exact_sql_deduplication"]["source_representation_count"] == 3
    assert inventory["exact_sql_deduplication"]["duplicate_representation_count"] == 1
    assert inventory["fixture_id"] == "imdb"
    assert "database" not in inventory
    assert "job_leakage_audit" not in inventory
    assert not (provenance / "job-overlap.json").exists()
    assert not (tmp_path / "job").exists()
