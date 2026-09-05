import io
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl.util.hashing import sha256_file
from scripts.shared import source_archive


def test_cached_download_is_verified_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"archive")
    specification = {"bytes": 7, "sha256": sha256_file(archive)}
    network = Mock(side_effect=AssertionError("cached archive must not be downloaded"))
    monkeypatch.setattr(source_archive.urllib.request, "urlopen", network)

    source_archive.download("https://example.invalid/archive", archive, specification)
    archive.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        source_archive.download(
            "https://example.invalid/archive", archive, specification
        )
    network.assert_not_called()


@pytest.mark.parametrize("member", ["../outside", "/absolute", "root/../../outside"])
def test_source_paths_cannot_escape(member: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        source_archive.safe_member_path(member)


def test_extract_source_strips_the_upstream_root(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    contents = b"SELECT 1;"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("upstream/1a.sql")
        member.size = len(contents)
        output.addfile(member, io.BytesIO(contents))
    target = tmp_path / "source"
    source_archive.extract_source(archive, target)
    assert (target / "1a.sql").read_bytes() == contents
