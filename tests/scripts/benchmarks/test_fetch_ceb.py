import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.benchmarks import fetch_ceb


def test_fetch_uses_source_without_recovery_report(
    repository_root: Path, tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "manifest.json"
    config.write_bytes((repository_root / "benchmarks/ceb/manifest.json").read_bytes())
    manifest = json.loads(config.read_text())
    download = Mock()
    extract = Mock()
    monkeypatch.setattr(fetch_ceb, "download", download)
    monkeypatch.setattr(fetch_ceb, "extract", extract)
    raw = tmp_path / "raw"
    monkeypatch.setattr(
        sys, "argv", ["fetch_ceb", "--manifest", str(config), "--raw-dir", str(raw)]
    )

    fetch_ceb.main()

    archive = manifest["source"]["archive"]
    path = raw / archive["filename"]
    download.assert_called_once_with(
        archive["url"],
        path,
        expected_size_in_bytes=archive["bytes"],
        expected_checksum=archive["sha256"],
    )
    extract.assert_called_once_with(path, raw / "source", manifest)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        (None, None),
        ("members", "archive member count differs"),
        ("full", "archive full query count differs"),
        ("unique_plans", "archive unique_plans query count differs"),
        ("templates", "archive full template counts differ"),
    ],
)
def test_selected_members_checks_source_and_query_counts(
    repository_root: Path, mismatch: str | None, message: str | None
) -> None:
    manifest = json.loads(
        (repository_root / "benchmarks/ceb/manifest.json").read_text()
    )
    manifest["source"]["archive"].update(members=2, regular_files=2)
    queries = manifest["queries"]
    queries.update(count=1, templates={"1a": 1})
    queries["subsets"]["unique_plans"].update(count=1, templates={"1a": 1})
    if mismatch == "members":
        manifest["source"]["archive"]["members"] = 3
    elif mismatch == "full":
        queries["count"] = 2
    elif mismatch == "unique_plans":
        queries["subsets"]["unique_plans"]["count"] = 2
    elif mismatch == "templates":
        queries["templates"] = {"2a": 1}

    archive_bytes = io.BytesIO()
    paths = ["imdb/1a/query.pkl", "imdb-unique-plans/1a/query.pkl"]
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for path in paths:
            archive.addfile(tarfile.TarInfo(f"source/queries/{path}"))
    archive_bytes.seek(0)
    with tarfile.open(fileobj=archive_bytes, mode="r") as archive:
        if message is None:
            assert set(fetch_ceb.selected_members(archive, manifest)) == set(paths)
        else:
            with pytest.raises(RuntimeError, match=message):
                fetch_ceb.selected_members(archive, manifest)
