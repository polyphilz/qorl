import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.benchmarks import fetch_job


@pytest.mark.parametrize("workload", ["job", "ceb"])
def test_workload_manifest_sections(repository_root: Path, workload: str) -> None:
    manifest = json.loads(
        (repository_root / "benchmarks" / workload / "manifest.json").read_text()
    )
    assert set(manifest) == {
        "schema_version",
        "workload_id",
        "fixture_id",
        "description",
        "source",
        "queries",
        "preparation",
    }
    assert manifest["schema_version"] == 2
    assert manifest["fixture_id"] == "imdb"
    assert set(manifest["source"]) == {"repository_url", "commit", "archive"}
    assert manifest["queries"]["count"] > 0
    assert (repository_root / manifest["preparation"]["script"]).is_file()


@pytest.mark.parametrize("valid", [True, False])
def test_job_fetch_downloads_only_queries(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
    valid: bool,
) -> None:
    config = repository_root / "benchmarks/job/manifest.json"
    value = json.loads(config.read_text())
    assert "dataset" not in value
    assert "schema" not in value["source"]
    assert "indexes" not in value["source"]
    download = Mock()
    monkeypatch.setattr(fetch_job, "download", download)
    monkeypatch.setattr(fetch_job, "extract_source", Mock())
    raw = tmp_path / "job"
    (raw / "source").mkdir(parents=True)
    for path in (repository_root / "benchmarks/job/queries").glob("*.sql"):
        (raw / "source" / path.name).symlink_to(path)
    if not valid:
        (raw / "source" / "1a.sql").unlink()
    monkeypatch.setattr(
        sys, "argv", ["fetch_job", "--manifest", str(config), "--raw-dir", str(raw)]
    )

    if valid:
        fetch_job.main()
    else:
        with pytest.raises(RuntimeError, match="JOB query count or checksum"):
            fetch_job.main()
    archive = value["source"]["archive"]
    download.assert_called_once_with(archive["url"], raw / archive["filename"], archive)
