import subprocess
from pathlib import Path

import pytest

from qorl.db import container as container_module
from qorl.db.container import PostgresContainer
from qorl.db.exceptions import WorkerError
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import load_runtime_profile


@pytest.fixture
def container(repository_root: Path, tmp_path: Path, monkeypatch):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    profile = load_runtime_profile(
        repository_root, Path("docker/worker_pool/configs/000-poolconf-1x32")
    )
    result = PostgresContainer(
        DatabaseFixture(repository_root, archive),
        "test-imdb",
        profile,
        profile.workers[0],
    )
    monkeypatch.setattr(container_module, "validate_host_topology", lambda *_: None)
    return result


def test_create_restore_start_use_the_image_and_volume_from_compose(
    container, monkeypatch
):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        output = ""
        code = 0
        if command[:3] == ["docker", "volume", "inspect"]:
            code = 1
        elif "ps" in command and container.created:
            output = "container-id"
        elif command[:2] == ["docker", "inspect"]:
            if command[-1] == "{{.Image}}":
                output = "sha256:installed-image"
            elif "Config.Env" in command[-1]:
                output = "POSTGRES_DB=qorl\nPGDATA=/var/lib/postgresql/18/docker\n"
            else:
                output = "test-imdb_qorl-postgres-data"
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(container, "execute", execute)

    container.create()
    container.restore_archive()
    container.start()
    container.stop()
    container.close()

    assert container.image_id == "sha256:installed-image"
    assert container.pgdata_relative_path == "18/docker"
    restore = next(command for command in calls if command[:2] == ["docker", "run"])
    assert "--network=none" in restore
    assert "sha256:installed-image" in restore
    assert "test-imdb_qorl-postgres-data:/target" in restore
    assert restore[-2:] == ["imdb.tar.gz", "18/docker"]
    assert next(
        i for i, command in enumerate(calls) if "create" in command
    ) < calls.index(restore)
    assert calls.index(restore) < next(
        i for i, command in enumerate(calls) if "up" in command
    )
    assert calls[-1][-2:] == ["down", "--volumes"]
    assert not container.created


@pytest.mark.parametrize("existing", ["project", "volume"])
def test_create_refuses_existing_resources(container, monkeypatch, existing):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        output = (
            "existing-container" if existing == "project" and "ps" in command else ""
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(container, "execute", execute)

    with pytest.raises(WorkerError, match="already exists"):
        container.create()

    assert not container.created
    assert not any("create" in command or "down" in command for command in calls)


@pytest.mark.parametrize("operation", ["restore_archive", "start"])
def test_population_and_start_require_creation(container, operation):
    with pytest.raises(WorkerError, match="create the container"):
        getattr(container, operation)()
