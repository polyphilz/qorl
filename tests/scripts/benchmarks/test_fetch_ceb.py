import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from qorl.util.hashing import sha256_bytes
from scripts.benchmarks import fetch_ceb


def test_fetch_uses_source_without_recovery_report(
    repository_root: Path, tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "manifest.json"
    config.write_bytes((repository_root / "benchmarks/ceb/manifest.json").read_bytes())
    manifest = json.loads(config.read_text())
    operations = Mock()
    monkeypatch.setattr(fetch_ceb, "download", operations.download)
    monkeypatch.setattr(fetch_ceb, "verify_file", operations.verify_file)
    monkeypatch.setattr(fetch_ceb, "extract", operations.extract)
    raw = tmp_path / "raw"
    monkeypatch.setattr(
        sys, "argv", ["fetch_ceb", "--manifest", str(config), "--raw-dir", str(raw)]
    )

    fetch_ceb.main()

    archive = manifest["source"]["archive"]
    path = raw / archive["filename"]
    assert operations.mock_calls == [
        call.download(archive["url"], path),
        call.verify_file(
            path,
            expected_size_in_bytes=archive["bytes"],
            expected_checksum=archive["sha256"],
        ),
        call.extract(path, raw / "source", manifest),
    ]


@pytest.mark.parametrize("cached", [False, True], ids=["downloaded", "cached"])
@pytest.mark.parametrize("mismatch", ["bytes", "sha256"])
def test_ceb_fetch_rejects_invalid_archive(
    repository_root: Path, tmp_path: Path, monkeypatch, cached: bool, mismatch: str
) -> None:
    manifest = json.loads(
        (repository_root / "benchmarks/ceb/manifest.json").read_text()
    )
    contents = b"archive"
    upstream = tmp_path / "upstream.tgz"
    upstream.write_bytes(contents)
    archive = manifest["source"]["archive"]
    archive["url"] = upstream.as_uri()
    archive["bytes"] = len(contents) + (1 if mismatch == "bytes" else 0)
    archive["sha256"] = sha256_bytes(b"different" if mismatch == "sha256" else contents)
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps(manifest))
    if cached:
        (tmp_path / archive["filename"]).write_bytes(contents)
    extraction = Mock()
    monkeypatch.setattr(fetch_ceb, "extract", extraction)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_ceb", "--manifest", str(config), "--raw-dir", str(tmp_path)],
    )

    message = "Size mismatch" if mismatch == "bytes" else "SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=message):
        fetch_ceb.main()

    extraction.assert_not_called()
    assert (tmp_path / archive["filename"]).read_bytes() == contents


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
