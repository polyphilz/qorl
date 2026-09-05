from pathlib import Path

import pytest

from qorl.db.container import PostgresContainer
from scripts.imdb import restore_and_verify as restore


@pytest.mark.parametrize("verification_fails", [False, True])
def test_restore_compares_loaded_fingerprints_without_raw_inputs(
    repository_root: Path, tmp_path: Path, monkeypatch, verification_fails: bool
) -> None:
    (tmp_path / "docker").symlink_to(
        repository_root / "docker", target_is_directory=True
    )
    reference = tmp_path / "data/imdb-verification/loaded.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("loaded report")
    (tmp_path / "data/imdb.tar.gz").write_bytes(b"archive")
    calls = []
    for operation in ("create", "restore_archive", "start", "close"):
        monkeypatch.setattr(
            PostgresContainer,
            operation,
            lambda *_, operation=operation: calls.append(operation),
        )

    def verify(container, output, *, repository, compare_to):
        assert compare_to == reference
        assert output == reference.with_name("restored.json")
        assert repository == tmp_path
        calls.append("compare")
        if verification_fails:
            raise RuntimeError("fingerprint mismatch")

    monkeypatch.setattr(restore, "verify", verify)
    if verification_fails:
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            restore.restore_and_verify(tmp_path)
        assert calls == ["create", "restore_archive", "start", "compare"]
    else:
        restore.restore_and_verify(tmp_path)
        assert calls == ["create", "restore_archive", "start", "compare", "close"]
    assert not (tmp_path / "data/raw").exists()


def test_restore_requires_loaded_verification_before_creating_a_container(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/imdb.tar.gz").write_bytes(b"archive")
    with pytest.raises(RuntimeError, match="loaded database verification is missing"):
        restore.restore_and_verify(tmp_path)
