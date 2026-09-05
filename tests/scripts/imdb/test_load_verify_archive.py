from pathlib import Path

import pytest

from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import load_runtime_profile
from scripts.imdb import load_verify_archive as load


@pytest.mark.parametrize("verification_fails", [False, True])
def test_load_verifies_then_stops_and_archives_without_fetching_or_restoring(
    repository_root: Path, tmp_path: Path, monkeypatch, verification_fails: bool
) -> None:
    (tmp_path / "docker").symlink_to(
        repository_root / "docker", target_is_directory=True
    )
    (tmp_path / "scripts").symlink_to(
        repository_root / "scripts", target_is_directory=True
    )
    (tmp_path / "data/raw/tables").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(load, "verify_inputs", lambda *_: calls.append("check-inputs"))
    for operation in ("create", "start", "stop", "close"):
        monkeypatch.setattr(
            PostgresContainer,
            operation,
            lambda *_, operation=operation: calls.append(operation),
        )
    monkeypatch.setattr(
        PostgresContainer,
        "restore_archive",
        lambda *_: pytest.fail("load must not restore"),
    )
    monkeypatch.setattr(
        load.PostgresWorker, "admin_sql", lambda _, sql: calls.append("load-sql")
    )

    def verify(*args, **kwargs):
        calls.append("verify")
        if verification_fails:
            raise RuntimeError("verification failed")

    monkeypatch.setattr(load, "verify", verify)
    monkeypatch.setattr(load, "archive_database", lambda *_: calls.append("archive"))

    if verification_fails:
        with pytest.raises(RuntimeError, match="verification failed"):
            load.load_verify_archive(tmp_path)
        assert calls == ["check-inputs", "create", "start", "load-sql", "verify"]
    else:
        load.load_verify_archive(tmp_path)
        assert calls == [
            "check-inputs",
            "create",
            "start",
            "load-sql",
            "verify",
            "stop",
            "archive",
            "close",
        ]


def test_load_requires_fetched_inputs_and_preserves_existing_archive(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="run fetch_imdb first"):
        load.load_verify_archive(tmp_path)
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        load.load_verify_archive(tmp_path)
    assert archive.read_bytes() == b"original"


@pytest.mark.parametrize("state", ["false 0", "true 0", "false 1"])
def test_archive_requires_clean_shutdown_and_writes_no_manifest(
    repository_root: Path, tmp_path: Path, monkeypatch, state: str
) -> None:
    profile = load_runtime_profile(repository_root, load.POOL_CONFIG)
    archive = tmp_path / "imdb.tar.gz"
    container = PostgresContainer(
        DatabaseFixture(repository_root, archive),
        "test-archive",
        profile,
        profile.workers[0],
    )
    container.container = "container"
    container.volume = "volume"
    container.image_id = "sha256:image"
    container.pgdata_relative_path = "18/docker"
    calls = []

    def command(arguments):
        calls.append(arguments)
        if arguments[:2] == ["docker", "inspect"]:
            return state
        assert "volume:/source:ro" in arguments
        assert "--network=none" in arguments
        assert arguments[-1] == "18/docker"
        (tmp_path / ".imdb.tar.gz.part").write_bytes(b"prepared archive")
        return ""

    monkeypatch.setattr(container, "command", command)
    if state != "false 0":
        with pytest.raises(RuntimeError, match="stopped cleanly"):
            load.archive_database(container, archive)
        assert not archive.exists()
        assert len(calls) == 1
    else:
        load.archive_database(container, archive)
        assert archive.read_bytes() == b"prepared archive"
        assert list(tmp_path.iterdir()) == [archive]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            load.archive_database(container, archive)
