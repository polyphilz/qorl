import io
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

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
    download = Mock()
    extract_dataset = Mock()
    monkeypatch.setattr(fetch, "download", download)
    monkeypatch.setattr(fetch, "extract_dataset", extract_dataset)
    raw = tmp_path / "imdb"
    monkeypatch.setattr(fetch, "IMDB_RAW_DATA_PATH", raw)
    fetch.main()

    download.assert_called_once_with(
        manifest.dataset.source_url,
        raw / manifest.dataset.archive.filename,
        expected_size_in_bytes=manifest.dataset.archive.bytes,
        expected_checksum=manifest.dataset.archive.sha256,
        print_progress=True,
        progress_interval_in_kib=128_000,
    )
    extract_dataset.assert_called_once_with(
        raw / manifest.dataset.archive.filename,
        raw / "tables",
    )


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
