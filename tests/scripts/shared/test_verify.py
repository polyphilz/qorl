from pathlib import Path

import pytest

from qorl.util.hashing import sha256_bytes
from scripts.shared.verify import verify_file


@pytest.mark.parametrize("contents", [b"archive", b""], ids=["nonempty", "empty"])
def test_verify_file_accepts_matching_contents(tmp_path: Path, contents: bytes) -> None:
    target = tmp_path / "archive.tgz"
    target.write_bytes(contents)

    verify_file(target, len(contents), sha256_bytes(contents))

    assert target.read_bytes() == contents


@pytest.mark.parametrize("mismatch", ["size", "checksum"])
def test_verify_file_reports_expected_and_actual_values(
    tmp_path: Path, mismatch: str
) -> None:
    target = tmp_path / "archive.tgz"
    contents = b"archive"
    target.write_bytes(contents)
    expected_size = len(contents) + (1 if mismatch == "size" else 0)
    expected_checksum = sha256_bytes(
        b"different" if mismatch == "checksum" else contents
    )
    label = "Size mismatch" if mismatch == "size" else "SHA-256 mismatch"

    with pytest.raises(RuntimeError, match=label) as error:
        verify_file(target, expected_size, expected_checksum)

    message = str(error.value)
    assert str(target) in message
    if mismatch == "size":
        assert f"expected {expected_size} bytes" in message
        assert f"got {len(contents)} bytes" in message
    else:
        assert f"expected {expected_checksum}" in message
        assert f"got {sha256_bytes(contents)}" in message
    assert target.read_bytes() == contents


@pytest.mark.parametrize("directory", [False, True], ids=["missing", "directory"])
def test_verify_file_requires_a_regular_file(tmp_path: Path, directory: bool) -> None:
    target = tmp_path / "archive.tgz"
    if directory:
        target.mkdir()

    with pytest.raises(RuntimeError, match="Expected a regular file") as error:
        verify_file(target, 0, sha256_bytes(b""))

    assert str(target) in str(error.value)
    assert target.exists() == directory
