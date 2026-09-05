import json
import sys
from pathlib import Path
from unittest.mock import Mock

from scripts.imdb import fetch_imdb


def test_fixture_fetch_uses_only_dataset_and_schema_inputs(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = repository_root / "scripts/imdb/imdb-metadata.json"
    value = json.loads(config.read_text())
    assert "workload" not in value
    assert "queries" not in value["schema_source"]
    download = Mock()
    extract_dataset = Mock()
    extract_source = Mock()
    verify_schema = Mock()
    monkeypatch.setattr(fetch_imdb, "download", download)
    monkeypatch.setattr(fetch_imdb, "extract_dataset", extract_dataset)
    monkeypatch.setattr(fetch_imdb, "extract_source", extract_source)
    monkeypatch.setattr(fetch_imdb, "verify_schema_directory", verify_schema)
    raw, source = tmp_path / "imdb", tmp_path / "job"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_imdb",
            "--manifest",
            str(config),
            "--raw-dir",
            str(raw),
            "--source-dir",
            str(source),
        ],
    )

    fetch_imdb.main()

    assert download.call_count == 2
    assert download.call_args_list[0].args[0] == value["dataset"]["source_url"]
    assert (
        download.call_args_list[1].args[1]
        == source / value["schema_source"]["archive"]["filename"]
    )
    extract_dataset.assert_called_once_with(
        raw / value["dataset"]["archive"]["filename"],
        raw / "tables",
        value["dataset"]["members"],
    )
    extract_source.assert_called_once()
    verify_schema.assert_called_once_with(source / "source", value["schema_source"])
