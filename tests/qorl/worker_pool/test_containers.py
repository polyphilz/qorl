import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl.plans.catalog import TaskCatalog
from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool import containers
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool, start_pool
from qorl.worker_pool.exceptions import ContainerError


@pytest.fixture(params=["000-poolconf-1x32", "001-poolconf-2x16", "002-poolconf-4x8"])
def pool(
    repository_root: Path, request, monkeypatch, postgres_config: PostgresConfig
) -> ContainerPool:
    config = load_pool_config(
        repository_root, Path("docker/worker_pool/configs") / request.param
    )
    monkeypatch.setattr(containers, "validate_host_topology", lambda *_: None)
    return ContainerPool(repository_root, "test-imdb", config, postgres_config)


@pytest.fixture
def docker_commands(
    pool: ContainerPool, monkeypatch, postgres_indexes: PostgresIndexes
):
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        output = ""
        code = 0
        if command[:3] == ["docker", "volume", "inspect"]:
            code = 1
        elif command[:2] == ["docker", "compose"]:
            slot = next(
                slot for slot in pool.workers if slot.compose_project_name == command[3]
            )
            if "ps" in command and slot.created:
                output = f"container-{slot.resources.index}"
        elif command[:2] == ["docker", "inspect"]:
            slot = pool.workers[int(command[2].removeprefix("container-"))]
            if command[-1] == "{{.Image}}":
                output = "sha256:installed-image"
            elif "Config.Env" in command[-1]:
                output = "POSTGRES_DB=qorl\nPGDATA=/var/lib/postgresql/18/docker\n"
            else:
                output = f"{slot.compose_project_name}_qorl-postgres-data"
        elif "pg_indexes" in (kwargs.get("input_text") or ""):
            output = postgres_indexes.model_dump_json()
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(pool, "execute", execute)
    return calls


def test_create_restore_start_stop_and_close(pool, docker_commands, tmp_path: Path):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")

    pool.create()
    pool.restore(archive)
    pool.start()
    pool.stop()
    pool.close()

    commands = [command for command, _ in docker_commands]
    restores = [command for command in commands if command[:2] == ["docker", "run"]]
    assert len(restores) == len(pool.workers)
    for slot, restore in zip(pool.workers, restores, strict=True):
        assert slot.image_id == "sha256:installed-image"
        assert slot.pgdata_relative_path == "18/docker"
        assert "--network=none" in restore
        assert "sha256:installed-image" in restore
        assert f"{slot.compose_project_name}_qorl-postgres-data:/target" in restore
        assert restore[-2:] == ["imdb.tar.gz", "18/docker"]
        assert not slot.created
        assert not slot.container_id
    assert max(
        i for i, command in enumerate(commands) if "create" in command
    ) < commands.index(restores[0])
    assert commands.index(restores[-1]) < next(
        i for i, command in enumerate(commands) if "up" in command
    )
    closes = [command for command in commands if command[-2:] == ["down", "--volumes"]]
    assert [command[3] for command in closes] == [
        slot.compose_project_name for slot in reversed(pool.workers)
    ]


def test_compose_receives_only_the_selected_configs(pool, docker_commands):
    pool.create()
    for slot in pool.workers:
        command, kwargs = next(
            (command, kwargs)
            for command, kwargs in docker_commands
            if command[:2] == ["docker", "compose"]
            and command[3] == slot.compose_project_name
        )
        resources = slot.resources
        environment = kwargs["environment"]
        assert environment["QORL_POSTGRES_CPUSET"] == resources.cpuset
        assert environment["QORL_POSTGRES_CPUSET_MEMS"] == resources.cpuset_mems
        assert environment["QORL_POSTGRES_MEMORY_LIMIT"] == resources.memory_limit
        assert environment["QORL_POSTGRES_MEMORY_BYTES"] == str(resources.memory_bytes)
        assert environment["QORL_POSTGRES_MEMORY_SWAP_LIMIT"] == resources.memory_limit
        assert environment["QORL_POSTGRES_MEMORY_SWAP_BYTES"] == "0"
        assert environment["QORL_POSTGRES_SHM_SIZE"] == resources.shm_size
        assert environment["QORL_POSTGRES_SHM_BYTES"] == str(resources.shm_bytes)
        assert environment["QORL_POSTGRES_PORT"] == str(resources.port)
        assert environment["QORL_POSTGRES_CONFIG_FILE"] == str(
            pool.postgres_config.pg_conf_path
        )
        assert environment["QORL_POSTGRES_EXPECTED_FILE"] == str(
            pool.postgres_config.expected_path
        )
        assert command[5] == str(pool.repository / "compose.yaml")


def test_sql_client_executes_inside_its_assigned_container(pool, monkeypatch):
    slot = pool.workers[0]
    slot.container_id = "assigned-container"
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, "query-result", ""))
    monkeypatch.setattr(pool, "execute", execute)

    assert slot.client.admin_sql("SELECT 1;") == "query-result"

    command = execute.call_args.args[0]
    assert command[:4] == ["docker", "exec", "--interactive", "assigned-container"]
    assert command[4:8] == ["bash", "-Eeuo", "pipefail", "-c"]
    assert execute.call_args.kwargs == {"input_text": "SELECT 1;", "check": False}


def test_clients_share_the_pools_settings(pool: ContainerPool) -> None:
    assert pool.settings == pool.postgres_config.agent_settings
    assert all(slot.client.settings is pool.settings for slot in pool.workers)


def test_start_pool_loads_one_shared_index_catalog(
    pool: ContainerPool,
    docker_commands,
    postgres_indexes: PostgresIndexes,
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(containers, "ContainerPool", Mock(return_value=pool))

    started = start_pool(
        pool.repository,
        "test",
        archive,
        postgres_config=pool.postgres_config,
        pool_config=pool.pool_config,
    )
    assert started.indexes == postgres_indexes
    assert all(slot.client.indexes is started.indexes for slot in started.workers)
    queries = [
        (position, command)
        for position, (command, options) in enumerate(docker_commands)
        if "pg_indexes" in (options.get("input_text") or "")
    ]
    assert len(queries) == 1
    position, command = queries[0]
    assert command[:4] == ["docker", "exec", "--interactive", "container-0"]
    assert position > max(
        i
        for i, (command, _) in enumerate(docker_commands)
        if command[-1] == "qorl-assert-config"
    )
    task = TaskSet.load(pool.repository, "job").tasks[0]
    command_count = len(docker_commands)
    for slot in started.workers:
        assert TaskCatalog.from_postgres(task, slot.client.indexes).indexes[
            "t"
        ] == frozenset({"title_pkey"})
    assert len(docker_commands) == command_count
    started.close()


@pytest.mark.parametrize("existing", ["project", "volume"])
def test_create_refuses_existing_resources(pool, monkeypatch, existing):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        output = (
            "existing-container" if existing == "project" and "ps" in command else ""
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(pool, "execute", execute)
    with pytest.raises(ContainerError, match="already exists"):
        pool.create()
    pool.close()
    assert not any(slot.created for slot in pool.workers)
    assert not any("create" in command or "down" in command for command in calls)


def test_population_and_start_require_creation(pool, tmp_path: Path):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    with pytest.raises(ContainerError, match="create the container"):
        pool.restore(archive)
    with pytest.raises(ContainerError, match="create the container"):
        pool.start()


def test_missing_archive_is_reported_before_creating_containers(
    pool, tmp_path: Path, monkeypatch
):
    create = Mock()
    monkeypatch.setattr(ContainerPool, "create", create)
    with pytest.raises(ContainerError, match="database archive is missing"):
        start_pool(
            pool.repository,
            "missing",
            tmp_path / "missing.tar.gz",
            pool_config=pool.pool_config,
            postgres_config=pool.postgres_config,
        )
    create.assert_not_called()


def test_claim_returns_workers_after_success_and_failure(pool):
    for expected in pool.workers:
        with pool.claim_worker() as slot:
            assert slot is expected
    with pytest.raises(ValueError, match="task failed"), pool.claim_worker() as first:
        raise ValueError("task failed")
    for _ in pool.workers[1:]:
        with pool.claim_worker():
            pass
    with pool.claim_worker() as returned:
        assert returned is first
    pool.close()


@pytest.mark.parametrize("phase", ["create", "restore", "start", "load_indexes"])
def test_startup_failure_cleans_owned_resources(
    pool, tmp_path: Path, monkeypatch, phase: str
):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    closed = []

    def create(self):
        for slot in self.workers:
            slot.created = True
        if phase == "create":
            raise ContainerError("startup failed")

    def restore(self, archive):
        if phase == "restore":
            raise ContainerError("startup failed")

    def start(self):
        if phase == "start":
            raise ContainerError("startup failed")

    def load_indexes(self):
        raise ContainerError("startup failed")

    def compose(self, slot, *args, **kwargs):
        assert args == ("down", "--volumes")
        closed.append(slot)
        return ""

    monkeypatch.setattr(ContainerPool, "create", create)
    monkeypatch.setattr(ContainerPool, "restore", restore)
    monkeypatch.setattr(ContainerPool, "start", start)
    monkeypatch.setattr(ContainerPool, "load_indexes", load_indexes)
    monkeypatch.setattr(ContainerPool, "compose_command", compose)
    with pytest.raises(ContainerError, match="startup failed"):
        start_pool(
            pool.repository,
            "failure",
            archive,
            pool_config=pool.pool_config,
            postgres_config=pool.postgres_config,
        )
    assert [slot.resources.index for slot in closed] == list(
        reversed(range(len(pool.workers)))
    )
    assert not any(slot.created for slot in closed)


def test_manifest_records_selected_config(pool):
    manifest = pool.manifest()
    assert manifest["worker_count"] == len(pool.workers)
    assert manifest["id"] == pool.pool_config.profile_id
    assert manifest["workers"] == [slot.resources.manifest() for slot in pool.workers]
    assert manifest["postgres_config"] == pool.postgres_config.manifest().model_dump()
