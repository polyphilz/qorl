import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock

import pytest

from qorl.paths import REPOSITORY_ROOT
from qorl.plans.catalog import TaskCatalog
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import PostgresIndexes
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool import containers
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool, start_pool
from qorl.worker_pool.exceptions import ContainerError
from qorl.worker_pool.schemas import ComposeEnvironment, PoolManifest, WorkerSlot


@pytest.fixture(params=["000-poolconf-1x32", "001-poolconf-2x16", "002-poolconf-4x8"])
def pool(
    repository_root: Path, request, monkeypatch, postgres_config: PostgresConfig
) -> ContainerPool:
    config = load_pool_config(Path("docker/worker_pool/configs") / request.param)
    monkeypatch.setattr(containers, "validate_host_topology", lambda *_: None)
    return ContainerPool("test-imdb", config, postgres_config)


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
        elif "pg_indexes" in (kwargs.get("stdin") or ""):
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
    for slot in pool.workers:
        restore = next(
            command for command in restores if f"{slot.volume}:/target" in command
        )
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
        assert isinstance(environment, ComposeEnvironment)
        assert environment.cpuset == resources.cpuset
        assert environment.cpuset_mems == resources.cpuset_mems
        assert environment.memory_limit == resources.memory_limit
        assert environment.memory_bytes == resources.memory_bytes
        assert environment.memory_swap_limit == resources.memory_limit
        assert environment.memory_swap_bytes == 0
        assert environment.shm_size == resources.shm_size
        assert environment.shm_bytes == resources.shm_bytes
        assert environment.port == resources.port
        assert environment.config_file == pool.postgres_config.pg_conf_path
        assert environment.expected_file == pool.postgres_config.expected_path
        assert command[5] == str(REPOSITORY_ROOT / "compose.yaml")


def test_start_updates_extension_before_waiting_for_config_health(
    pool, docker_commands
):
    pool.create()
    pool.start()

    for slot in pool.workers:
        assert slot.client.allocation is not None
        assert slot.client.allocation.cpuset == slot.resources.cpuset
        assert slot.client.allocation.memory_bytes == slot.resources.memory_bytes
        assert (
            slot.client.allocation.physical_core_count
            == slot.resources.physical_core_count
        )
        startup = [
            (command, options)
            for command, options in docker_commands
            if slot.container_id in command or slot.compose_project_name in command
        ]
        ready = next(
            i for i, (command, _) in enumerate(startup) if "pg_isready" in command
        )
        update = next(
            i
            for i, (_, options) in enumerate(startup)
            if options.get("stdin") == "ALTER EXTENSION pg_hint_plan UPDATE;"
        )
        health = next(
            i for i, (command, _) in enumerate(startup) if "--wait" in command
        )
        check = next(
            i
            for i, (command, _) in enumerate(startup)
            if command[-1] == "qorl-assert-config"
        )
        assert ready < update < health < check
        assert "--host=127.0.0.1" in startup[ready][0]


def test_start_polls_until_postgres_accepts_tcp_connections(pool, monkeypatch):
    compose = Mock()
    not_ready = subprocess.CompletedProcess([], 1, "", "")
    ready = subprocess.CompletedProcess([], 0, "", "")
    execute = Mock(side_effect=[not_ready, ready, ready])
    update = Mock()
    sleep = Mock()
    slot = pool.workers[0]
    monkeypatch.setattr(pool, "compose_command", compose)
    monkeypatch.setattr(pool, "execute", execute)
    monkeypatch.setattr(slot.client, "admin_sql", update)
    monkeypatch.setattr(containers.time, "sleep", sleep)

    pool._start(slot)

    sleep.assert_called_once_with(containers.STARTUP_POLL_SECONDS)
    update.assert_called_once_with("ALTER EXTENSION pg_hint_plan UPDATE;")
    assert execute.call_args_list[0].args == execute.call_args_list[1].args
    assert execute.call_args.args[0][-1] == "qorl-assert-config"


def test_startup_timeout_does_not_attempt_extension_update(pool, monkeypatch):
    compose = Mock()
    execute = Mock(return_value=subprocess.CompletedProcess([], 1, "", ""))
    update = Mock()
    slot = pool.workers[0]
    monkeypatch.setattr(pool, "compose_command", compose)
    monkeypatch.setattr(pool, "execute", execute)
    monkeypatch.setattr(slot.client, "admin_sql", update)
    monkeypatch.setattr(
        containers.time,
        "monotonic",
        Mock(side_effect=[0, containers.STARTUP_TIMEOUT_SECONDS]),
    )

    with pytest.raises(ContainerError, match="PostgreSQL startup timed out"):
        pool._start(slot)

    update.assert_not_called()
    compose.assert_called_once_with(slot, ["up", "--detach", "--no-build", "postgres"])


def test_extension_update_failure_cleans_owned_workers(
    pool, docker_commands, tmp_path, monkeypatch
):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    execute = pool.execute

    def fail_update(command, **kwargs):
        if kwargs.get("stdin") == "ALTER EXTENSION pg_hint_plan UPDATE;":
            return subprocess.CompletedProcess(
                command, 1, "", "extension update failed"
            )
        return execute(command, **kwargs)

    monkeypatch.setattr(pool, "execute", fail_update)
    monkeypatch.setattr(containers, "ContainerPool", Mock(return_value=pool))
    with pytest.raises(PostgresError, match="extension update failed"):
        start_pool(
            "failed-update",
            archive,
            postgres_config=pool.postgres_config,
            pool_config=pool.pool_config,
        )

    assert not any(slot.created for slot in pool.workers)
    assert not any("--wait" in command for command, _ in docker_commands)


def test_sql_client_executes_inside_its_assigned_container(pool, monkeypatch):
    slot = pool.workers[0]
    slot.container_id = "assigned-container"
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, "query-result", ""))
    monkeypatch.setattr(pool, "execute", execute)

    assert slot.client.admin_sql("SELECT 1;") == "query-result"

    command = execute.call_args.args[0]
    assert command[:4] == ["docker", "exec", "--interactive", "assigned-container"]
    assert command[4:8] == ["bash", "-Eeuo", "pipefail", "-c"]
    assert execute.call_args.kwargs == {"stdin": "SELECT 1;", "check": False}


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
        if "pg_indexes" in (options.get("stdin") or "")
    ]
    assert len(queries) == 1
    position, command = queries[0]
    assert command[:4] == ["docker", "exec", "--interactive", "container-0"]
    assert position > max(
        i
        for i, (command, _) in enumerate(docker_commands)
        if command[-1] == "qorl-assert-config"
    )
    task = TaskSet.load(REPOSITORY_ROOT, "job").tasks[0]
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

    def compose(self, slot, arguments):
        assert arguments == ["down", "--volumes"]
        closed.append(slot)
        return ""

    monkeypatch.setattr(ContainerPool, "create", create)
    monkeypatch.setattr(ContainerPool, "restore", restore)
    monkeypatch.setattr(ContainerPool, "start", start)
    monkeypatch.setattr(ContainerPool, "load_indexes", load_indexes)
    monkeypatch.setattr(ContainerPool, "compose_command", compose)
    with pytest.raises(ContainerError, match="startup failed"):
        start_pool(
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
    assert isinstance(manifest, PoolManifest)
    assert manifest.worker_count == len(pool.workers)
    assert manifest.id == pool.pool_config.profile_id
    assert manifest.workers == [slot.resources.manifest() for slot in pool.workers]
    assert manifest.postgres_config == pool.postgres_config.manifest()
    assert manifest.model_dump() == {
        "id": pool.pool_config.profile_id,
        "path": str(pool.pool_config.path),
        "config_sha256": pool.pool_config.sha256,
        "worker_count": len(pool.workers),
        "workers": [slot.resources.manifest().model_dump() for slot in pool.workers],
        "postgres_config": pool.postgres_config.manifest().model_dump(),
    }


@pytest.mark.parametrize("phase", ["create", "restore", "start"])
def test_startup_stages_run_workers_concurrently(pool, tmp_path, monkeypatch, phase):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    rendezvous = Barrier(len(pool.workers), timeout=5)
    visited: list[int] = []
    lock = Lock()
    for slot in pool.workers:
        slot.created = True

    def operation(slot: WorkerSlot, archive: Path | None = None) -> None:
        if phase == "restore":
            assert archive == tmp_path / "imdb.tar.gz"
        with lock:
            visited.append(slot.resources.index)
        rendezvous.wait()

    monkeypatch.setattr(pool, f"_{phase}", operation)
    if phase == "restore":
        pool.restore(archive)
    else:
        getattr(pool, phase)()
    assert sorted(visited) == [slot.resources.index for slot in pool.workers]


@pytest.mark.parametrize("phase", ["create", "restore", "start"])
def test_startup_joins_active_operations_before_cleanup(
    pool_config, postgres_config, tmp_path, monkeypatch, phase
):
    pool = ContainerPool("concurrent-failure", pool_config, postgres_config)
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")
    rendezvous = Barrier(len(pool.workers), timeout=5)
    failed = Event()
    release = Event()
    cleanup_started = Event()
    finished = [Event() for _ in pool.workers]
    cleaned: list[int] = []

    def operation(slot: WorkerSlot, archive: Path | None = None) -> None:
        slot.created = True
        try:
            rendezvous.wait()
            if slot.resources.index == 0:
                failed.set()
                raise ContainerError("startup failed")
            assert release.wait(5)
        finally:
            finished[slot.resources.index].set()

    def create() -> None:
        for slot in pool.workers:
            slot.created = True

    def compose(slot: WorkerSlot, arguments: list[str]) -> str:
        assert arguments == ["down", "--volumes"]
        cleanup_started.set()
        assert all(done.is_set() for done in finished)
        cleaned.append(slot.resources.index)
        return ""

    monkeypatch.setattr(containers, "validate_host_topology", lambda *_: None)
    monkeypatch.setattr(containers, "ContainerPool", Mock(return_value=pool))
    if phase != "create":
        monkeypatch.setattr(pool, "create", create)
    if phase == "start":
        monkeypatch.setattr(pool, "restore", lambda _: None)
    monkeypatch.setattr(pool, f"_{phase}", operation)
    monkeypatch.setattr(pool, "compose_command", compose)
    with ThreadPoolExecutor(max_workers=1) as executor:
        startup = executor.submit(
            start_pool,
            "concurrent-failure",
            archive,
            postgres_config=postgres_config,
            pool_config=pool_config,
        )
        try:
            assert failed.wait(5)
            assert not cleanup_started.wait(0.05)
            assert not startup.done()
        finally:
            release.set()
        with pytest.raises(ContainerError, match="startup failed"):
            startup.result(timeout=5)
    assert cleaned == list(reversed(range(len(pool.workers))))
    assert not any(slot.created for slot in pool.workers)


def test_parallel_failure_cancels_pending_work(pool, monkeypatch):
    futures: list[Future[None]] = [Future() for _ in pool.workers]
    futures[0].set_exception(ContainerError("operation failed"))
    executor = Mock()
    executor.__enter__ = Mock(return_value=executor)
    executor.__exit__ = Mock(return_value=False)
    executor.submit.side_effect = futures
    monkeypatch.setattr(containers, "ThreadPoolExecutor", Mock(return_value=executor))

    with pytest.raises(ContainerError, match="operation failed"):
        pool._parallel(Mock())

    assert all(future.cancelled() for future in futures[1:])
    executor.__exit__.assert_called_once()


def test_failed_create_still_cleans_partial_resources(
    pool, docker_commands, monkeypatch
):
    execute = pool.execute
    failed_slot = pool.workers[0]

    def fail_create(command, **kwargs):
        if "create" in command and command[3] == failed_slot.compose_project_name:
            raise ContainerError("partial create")
        return execute(command, **kwargs)

    monkeypatch.setattr(pool, "execute", fail_create)
    with pytest.raises(ContainerError, match="partial create"):
        pool.create()
    assert failed_slot.created
    pool.close()
    assert any(
        command[3] == failed_slot.compose_project_name
        for command, _ in docker_commands
        if command[-2:] == ["down", "--volumes"]
    )
    assert not any(slot.created for slot in pool.workers)


@pytest.mark.parametrize("error_type", [ContainerError, OSError])
def test_cleanup_continues_after_failure_and_retains_worker_for_retry(
    pool, monkeypatch, caplog, error_type
):
    for slot in pool.workers:
        slot.created = True
        slot.container_id = f"container-{slot.resources.index}"
    failed_slot = pool.workers[-1]
    visited: list[int] = []

    def compose(slot: WorkerSlot, arguments: list[str]) -> str:
        assert arguments == ["down", "--volumes"]
        visited.append(slot.resources.index)
        if len(visited) == 1:
            raise error_type("cannot remove container")
        return ""

    monkeypatch.setattr(pool, "compose_command", compose)
    pool.close()
    assert visited == [slot.resources.index for slot in reversed(pool.workers)]
    assert failed_slot.created
    assert failed_slot.container_id
    assert all(not slot.created for slot in pool.workers[:-1])
    assert failed_slot.compose_project_name in caplog.text
    assert "cannot remove container" in caplog.text

    pool.close()
    assert visited[-1] == failed_slot.resources.index
    assert len(visited) == len(pool.workers) + 1
    assert not any(slot.created for slot in pool.workers)


def test_cleanup_failure_does_not_replace_startup_error(
    pool, tmp_path, monkeypatch, caplog
):
    archive = tmp_path / "imdb.tar.gz"
    archive.write_bytes(b"archive")

    def create() -> None:
        pool.workers[0].created = True
        raise ContainerError("original startup failure")

    monkeypatch.setattr(containers, "ContainerPool", Mock(return_value=pool))
    monkeypatch.setattr(pool, "create", create)
    monkeypatch.setattr(
        pool, "compose_command", Mock(side_effect=ContainerError("cleanup failure"))
    )
    with pytest.raises(ContainerError, match="original startup failure"):
        start_pool(
            "failed",
            archive,
            postgres_config=pool.postgres_config,
            pool_config=pool.pool_config,
        )
    assert "cleanup failure" in caplog.text
    assert pool.workers[0].created


@pytest.mark.parametrize("stdin", [None, "SELECT 1;"])
def test_execute_preserves_input_and_inherits_environment(
    pool, monkeypatch, tmp_path, stdin
):
    completed = subprocess.CompletedProcess(["docker", "info"], 0, "output", "")
    run = Mock(return_value=completed)
    monkeypatch.setattr(containers.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)

    assert pool.execute(["docker", "info"], stdin=stdin) is completed
    run.assert_called_once_with(
        ["docker", "info"],
        cwd=REPOSITORY_ROOT,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=None,
    )


def test_compose_environment_overrides_and_preserves_os_variables(pool, monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "output", ""))
    monkeypatch.setattr(containers.subprocess, "run", run)
    monkeypatch.setenv("QORL_POOL_TEST_SENTINEL", "inherited")
    monkeypatch.setenv("QORL_POSTGRES_PORT", "wrong")
    slot = pool.workers[0]

    assert pool.compose_command(slot, ["config", "--quiet"]) == "output"

    environment = run.call_args.kwargs["env"]
    assert environment["QORL_POOL_TEST_SENTINEL"] == "inherited"
    assert environment["QORL_POSTGRES_PORT"] == str(slot.resources.port)
    assert environment["QORL_POSTGRES_CONFIG_FILE"] == str(
        pool.postgres_config.pg_conf_path
    )
    assert all(isinstance(value, str) for value in environment.values())


@pytest.mark.parametrize("stderr", ["error detail", ""])
def test_execute_failure_is_checked_unless_explicitly_disabled(
    pool, monkeypatch, stderr
):
    completed = subprocess.CompletedProcess([], 1, "stdout detail", stderr)
    monkeypatch.setattr(containers.subprocess, "run", Mock(return_value=completed))

    with pytest.raises(ContainerError, match=stderr or "stdout detail"):
        pool.execute(["docker", "info"])
    assert pool.execute(["docker", "info"], check=False) is completed
    with pytest.raises(ContainerError, match=stderr or "stdout detail"):
        pool.compose_command(pool.workers[0], ["down", "--volumes"])
