import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from qorl.util.hashing import sha256_bytes
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
    operations = Mock()
    monkeypatch.setattr(fetch_job, "download", operations.download)
    monkeypatch.setattr(fetch_job, "verify_file", operations.verify_file)
    monkeypatch.setattr(fetch_job, "extract_source", operations.extract_source)
    raw = tmp_path / "job"
    (raw / "source").mkdir(parents=True)
    archive = value["source"]["archive"]
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
    path = raw / archive["filename"]
    assert operations.mock_calls == [
        call.download(archive["url"], path),
        call.verify_file(
            path,
            expected_size_in_bytes=archive["bytes"],
            expected_checksum=archive["sha256"],
        ),
        call.extract_source(path, raw / "source"),
    ]


@pytest.mark.parametrize("cached", [False, True], ids=["downloaded", "cached"])
@pytest.mark.parametrize("mismatch", ["bytes", "sha256"])
def test_job_fetch_rejects_invalid_archive(
    repository_root: Path, tmp_path: Path, monkeypatch, cached: bool, mismatch: str
) -> None:
    manifest = json.loads(
        (repository_root / "benchmarks/job/manifest.json").read_text()
    )
    contents = b"archive"
    upstream = tmp_path / "upstream.tgz"
    upstream.write_bytes(contents)
    archive = manifest["source"]["archive"]
    archive["url"] = upstream.as_uri()
    archive["bytes"] = len(contents)
    archive["sha256"] = sha256_bytes(contents)
    if mismatch == "bytes":
        archive["bytes"] += 1
    else:
        archive["sha256"] = sha256_bytes(b"different")
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps(manifest))
    if cached:
        (tmp_path / archive["filename"]).write_bytes(contents)
    extraction = Mock()
    monkeypatch.setattr(fetch_job, "extract_source", extraction)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_job", "--manifest", str(config), "--raw-dir", str(tmp_path)],
    )

    message = "Size mismatch" if mismatch == "bytes" else "SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=message):
        fetch_job.main()
    extraction.assert_not_called()
    assert (tmp_path / archive["filename"]).read_bytes() == contents


@pytest.mark.parametrize("member", ["../outside", "/absolute", "root/../../outside"])
def test_source_paths_cannot_escape(member: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        fetch_job.safe_member_path(member)


def test_extract_source_strips_the_upstream_root(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    contents = b"SELECT 1;"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("upstream/1a.sql")
        member.size = len(contents)
        output.addfile(member, io.BytesIO(contents))
    target = tmp_path / "source"
    fetch_job.extract_source(archive, target)
    assert (target / "1a.sql").read_bytes() == contents
