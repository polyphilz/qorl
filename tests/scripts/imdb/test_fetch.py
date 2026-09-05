import io
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from pydantic import ValidationError

from qorl.util.hashing import sha256_bytes
from scripts.imdb import fetch
from scripts.imdb.schemas import ImdbManifest


def test_module_import(repository_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-E", "-c", "import scripts.imdb.fetch"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_fixture_fetch_downloads_only_the_imdb_dataset(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = repository_root / "scripts/imdb/manifest.json"
    assert config == fetch.IMDB_MANIFEST
    assert repository_root / "data/raw" == fetch.IMDB_RAW_DATA_PATH
    manifest = ImdbManifest.model_validate_json(config.read_text())
    operations = Mock()
    monkeypatch.setattr(fetch, "download", operations.download)
    monkeypatch.setattr(fetch, "verify_file", operations.verify_file)
    monkeypatch.setattr(fetch, "extract_dataset", operations.extract_dataset)
    raw = tmp_path / "imdb"
    monkeypatch.setattr(fetch, "IMDB_RAW_DATA_PATH", raw)
    fetch.main()

    archive = raw / manifest.dataset.archive.filename
    assert operations.mock_calls == [
        call.download(
            manifest.dataset.source_url,
            archive,
            print_progress=True,
            progress_interval_in_kib=128_000,
        ),
        call.verify_file(
            archive,
            expected_size_in_bytes=manifest.dataset.archive.bytes,
            expected_checksum=manifest.dataset.archive.sha256,
        ),
        call.extract_dataset(archive, raw / "tables"),
    ]


@pytest.mark.parametrize("cached", [False, True], ids=["downloaded", "cached"])
@pytest.mark.parametrize("mismatch", ["bytes", "sha256"])
def test_invalid_archive_fails_before_extraction(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
    cached: bool,
    mismatch: str,
) -> None:
    manifest = ImdbManifest.model_validate_json(
        (repository_root / "scripts/imdb/manifest.json").read_text()
    )
    contents = b"archive"
    upstream = tmp_path / "upstream.tgz"
    upstream.write_bytes(contents)
    specification = manifest.dataset.archive.model_copy(
        update={
            "bytes": len(contents) + (1 if mismatch == "bytes" else 0),
            "sha256": sha256_bytes(b"different" if mismatch == "sha256" else contents),
        }
    )
    manifest = manifest.model_copy(
        update={
            "dataset": manifest.dataset.model_copy(
                update={"source_url": upstream.as_uri(), "archive": specification}
            )
        }
    )
    config = tmp_path / "manifest.json"
    config.write_text(manifest.model_dump_json())
    archive = tmp_path / specification.filename
    if cached:
        archive.write_bytes(contents)
    extraction = Mock()
    monkeypatch.setattr(fetch, "IMDB_MANIFEST", config)
    monkeypatch.setattr(fetch, "IMDB_RAW_DATA_PATH", tmp_path)
    monkeypatch.setattr(fetch, "extract_dataset", extraction)

    message = "Size mismatch" if mismatch == "bytes" else "SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=message):
        fetch.main()

    extraction.assert_not_called()
    assert archive.read_bytes() == contents


def test_invalid_manifest_fails_before_fetching(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "manifest.json"
    config.write_text('{"dataset": {"archive": {"bytes": true}}}')
    raw = tmp_path / "raw"
    download = Mock()
    monkeypatch.setattr(fetch, "IMDB_MANIFEST", config)
    monkeypatch.setattr(fetch, "IMDB_RAW_DATA_PATH", raw)
    monkeypatch.setattr(fetch, "download", download)

    with pytest.raises(ValidationError):
        fetch.main()
    download.assert_not_called()
    assert not raw.exists()


def test_extract_dataset_does_not_verify_contents(tmp_path: Path) -> None:
    archive = tmp_path / "imdb.tgz"
    contents = b"not a CSV"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("unexpected.txt")
        member.size = len(contents)
        output.addfile(member, io.BytesIO(contents))
    target = tmp_path / "tables"

    fetch.extract_dataset(archive, target)
    assert (target / "unexpected.txt").read_bytes() == contents

    (target / "unexpected.txt").write_bytes(b"modified")
    archive.unlink()
    fetch.extract_dataset(archive, target)
    assert (target / "unexpected.txt").read_bytes() == b"modified"


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_extract_dataset_rejects_unsafe_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "imdb.tgz"
    with tarfile.open(archive, "w:gz") as output:
        output.addfile(tarfile.TarInfo(name), io.BytesIO())
    target = tmp_path / "tables"

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        fetch.extract_dataset(archive, target)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".imdb.extract.*"))
