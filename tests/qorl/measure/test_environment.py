import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from qorl.measure import environment
from qorl.measure.environment import (
    capture_environment,
    run_json,
    safe_container_inspect,
)
from qorl.measure.schemas import CapturePhase, DockerHealth, EnvironmentCapture
from qorl.model.schemas import JsonObject
from qorl.postgres.config import PostgresConfig
from qorl.util.hashing import sha256_file
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig


@pytest.fixture
def inspect_state(monkeypatch: pytest.MonkeyPatch) -> JsonObject:
    state: JsonObject = {"Status": "running", "StartedAt": "2026-09-08T00:00:00Z"}
    raw: JsonObject = {
        "Id": "container-id",
        "Name": "/worker",
        "Created": "created",
        "Image": "image-id",
        "State": state,
        "Config": {"Image": "postgres:test", "Env": ["PASSWORD=private-value"]},
        "HostConfig": {
            "CpusetCpus": "0-3",
            "CpusetMems": "0",
            "CpuPeriod": 0,
            "CpuQuota": 0,
            "NanoCpus": 0,
            "Memory": 1024,
            "MemorySwap": 1024,
            "ShmSize": 512,
            "PidsLimit": None,
            "ReadonlyRootfs": False,
            "CgroupnsMode": "private",
        },
        "Mounts": [
            {
                "Type": "bind",
                "Destination": "/config",
                "RW": False,
                "Source": "/private-value",
            },
            {
                "Type": "volume",
                "Name": "data",
                "Destination": "/data",
                "Driver": "local",
                "Mode": "rw",
                "RW": True,
                "Propagation": "",
                "Source": "/private-value",
            },
        ],
    }

    def inspect(command: list[str]) -> list[JsonObject]:
        assert command == ["docker", "container", "inspect", "worker"]
        return [raw]

    monkeypatch.setattr("qorl.measure.environment.run_json", inspect)
    return state


@pytest.mark.parametrize(
    "present,health,expected",
    [
        (False, None, None),
        (True, None, None),
        (True, {}, None),
        (True, {"Status": None}, None),
        (True, {"Status": "healthy", "Log": [{"Output": "private-value"}]}, "healthy"),
        (True, {"Status": "unhealthy"}, "unhealthy"),
        (True, {"Status": "starting"}, "starting"),
    ],
)
def test_optional_health_preserves_safe_capture(
    inspect_state: JsonObject,
    present: bool,
    health: JsonValue,
    expected: str | None,
) -> None:
    if present:
        inspect_state["Health"] = health
    captured = safe_container_inspect("worker")
    assert (
        type(captured).model_validate_json(captured.model_dump_json(by_alias=True))
        == captured
    )
    assert captured.state.model_dump() == {
        "status": "running",
        "started_at": "2026-09-08T00:00:00Z",
        "health": expected,
    }
    assert set(captured.model_dump()) == {
        "id",
        "name",
        "created",
        "image_id",
        "image_reference",
        "labels",
        "state",
        "limits",
        "mounts",
    }
    assert "private-value" not in captured.model_dump_json()


@pytest.mark.parametrize("health", [{"Status": 42}, {"Status": {"nested": "value"}}])
def test_health_status_is_validated(
    inspect_state: JsonObject, health: JsonObject
) -> None:
    inspect_state["Health"] = health
    with pytest.raises(ValidationError, match="Status"):
        safe_container_inspect("worker")


@pytest.fixture
def host_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "/etc/os-release": 'NAME="Test Linux"\nID=test\n',
        "/proc/meminfo": "MemTotal: 100 kB\nHugePages_Total: 0\nIgnored: 42\n",
        "/proc/cpuinfo": "processor: 0\nmicrocode: 0xab\n",
        "/proc/cmdline": "quiet\n",
        "/sys/devices/system/cpu/isolated": "2-3\n",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "test-driver\n",
        "/sys/block/test/queue/scheduler": "none [test]\n",
        "/sys/devices/virtual/dmi/id/sys_vendor": "Test vendor\n",
    }
    root = tmp_path / "host"
    for name, content in files.items():
        path = root / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def host_path(value: str) -> Path:
        return (
            root / value.lstrip("/")
            if value.startswith(("/sys/", "/proc/", "/etc/"))
            else Path(value)
        )

    monkeypatch.setattr(environment, "Path", host_path)
    for name in ("node", "system", "release", "version", "machine"):
        monkeypatch.setattr(environment.platform, name, lambda: "test-host")


@pytest.fixture
def commands(
    inspect_state: JsonObject, monkeypatch: pytest.MonkeyPatch
) -> list[list[str]]:
    # Use the safe Docker fixture's external document, not its captured projection.
    container = environment.run_json(["docker", "container", "inspect", "worker"])
    monkeypatch.setattr(environment, "run_json", run_json)
    calls: list[list[str]] = []
    image: JsonObject = {
        "Id": "image-id",
        "Created": "created",
        "Os": "linux",
        "Architecture": "amd64",
        "Size": 123,
        "Config": {"Env": ["PRIVATE_IMAGE"]},
        "RootFS": {"Layers": None},
        "RepoTags": None,
    }

    def command(arguments: list[str], *, check: bool = True) -> str:
        calls.append(arguments)
        if arguments[:3] == ["docker", "container", "inspect"]:
            assert arguments[3] == "active-container"
            return json.dumps(container)
        if arguments[:3] == ["docker", "image", "inspect"]:
            assert arguments[3] == "image-id"
            return json.dumps([image])
        if arguments[:2] == ["docker", "exec"]:
            assert arguments[2] == "active-container"
            if arguments[3] == "/usr/local/bin/qorl-assert-config":
                return "OK\n"
            if arguments[3] == "/usr/local/bin/qorl-dump-postgres-state":
                return (
                    '{"postgres":"evidence"}\n'
                    if arguments[4].endswith("json")
                    else "name,value\nsetting,on\n"
                )
            if arguments[3] == "sh":
                return "cpus_allowed_list=0-3\nmems_allowed_list=0\ncpu_max=max 100000\nmemory_max=1024\nmemory_swap_max=0\nshm_size_bytes=512\n"
            assert arguments[3] == "sha256sum"
            return "".join(f"digest {name}\n" for name in arguments[4:])
        if arguments[:2] == ["docker", "info"]:
            return '{"NCPU":8,"LiveRestoreEnabled":false,"SecurityOptions":null,"Uncaptured":"private-value"}'
        if arguments[:2] == ["docker", "compose"]:
            return "compose-test\n"
        if arguments[0] == "nvidia-smi":
            assert not check
            return ""
        return '{"opaque":"command evidence"}'

    monkeypatch.setattr(environment, "run", command)
    return calls


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_direct_capture_complete_wire_and_artifacts(
    commands: list[list[str]],
    host_files: None,
    phase: CapturePhase,
    tmp_path: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ContainerPool("capture-test", pool_config, postgres_config)
    slot = pool.workers[-1]
    slot.container_id = "active-container"

    def no_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "capture must not launch a Python process or reload configs"
        )

    monkeypatch.setattr(pool, "execute", no_subprocess)
    monkeypatch.setattr(PostgresConfig, "load", no_subprocess)
    output = tmp_path / "evidence"
    capture_environment(pool, slot, output, phase)
    assert capsys.readouterr().out == ""
    path = output / ("environment.json" if phase == "pre" else "environment.post.json")
    captured = EnvironmentCapture.model_validate_json(path.read_bytes())
    assert (
        EnvironmentCapture.model_validate_json(captured.model_dump_json(by_alias=True))
        == captured
    )
    assert environment.RAW_COMMAND.validate_json(
        path.read_bytes()
    ) == environment.RAW_COMMAND.validate_json(captured.model_dump_json(by_alias=True))
    assert captured.phase == phase and captured.schema_version == 1
    assert captured.postgres_config == postgres_config.manifest()
    assert captured.runtime_profile.configuration == pool_config.configuration
    assert captured.runtime_profile.id == pool_config.profile_id
    assert captured.runtime_profile.sha256 == pool_config.sha256
    assert captured.runtime_profile.path == str(pool_config.path)
    assert captured.runtime_identity.postgres_image_id == "image-id"
    assert captured.runtime_identity.postgres_config_id == postgres_config.config_id
    assert captured.assertion_output == "OK"
    assert captured.host.model_dump() == {
        "hostname": "test-host",
        "os_release": {"NAME": "Test Linux", "ID": "test"},
        "kernel": {
            "system": "test-host",
            "release": "test-host",
            "version": "test-host",
            "machine": "test-host",
            "command_line": "quiet",
        },
        "machine": {
            "sys_vendor": "Test vendor",
            "product_name": None,
            "product_version": None,
            "bios_version": None,
        },
        "cpu": {
            "summary": {"opaque": "command evidence"},
            "topology": {"opaque": "command evidence"},
            "microcode": "0xab",
            "isolated": "2-3",
            "power_policies": [
                {
                    "policy": "policy0",
                    "affected_cpus": None,
                    "scaling_driver": "test-driver",
                    "scaling_governor": None,
                    "energy_performance_preference": None,
                    "cpuinfo_min_freq": None,
                    "cpuinfo_max_freq": None,
                }
            ],
        },
        "memory": {
            "meminfo": {"MemTotal": "100 kB", "HugePages_Total": "0"},
            "transparent_huge_pages": None,
        },
        "storage": {
            "docker_root_mount": {"opaque": "command evidence"},
            "block_devices": {"opaque": "command evidence"},
            "schedulers": {"test": "none [test]"},
        },
        "gpus": None,
    }
    assert captured.docker.image.model_dump() == {
        "id": "image-id",
        "repo_tags": [],
        "repo_digests": [],
        "created": "created",
        "os": "linux",
        "architecture": "amd64",
        "size": 123,
        "labels": {},
        "rootfs_layers": [],
    }
    info = captured.docker.info.model_dump(by_alias=True)
    assert info["NCPU"] == 8 and info["LiveRestoreEnabled"] is False
    assert info["SecurityOptions"] is None and info["ServerVersion"] is None
    assert len(info) == 15
    assert captured.docker.container_runtime.shm_size_bytes == "512"
    assert captured.docker.container.model_dump() == {
        "id": "container-id",
        "name": "worker",
        "created": "created",
        "image_id": "image-id",
        "image_reference": "postgres:test",
        "labels": {},
        "state": {
            "status": "running",
            "started_at": "2026-09-08T00:00:00Z",
            "health": None,
        },
        "limits": {
            "cpuset_cpus": "0-3",
            "cpuset_mems": "0",
            "cpu_period": 0,
            "cpu_quota": 0,
            "nano_cpus": 0,
            "memory": 1024,
            "memory_swap": 1024,
            "shm_size": 512,
            "pids_limit": None,
            "readonly_rootfs": False,
            "cgroupns_mode": "private",
        },
        "mounts": [
            {
                "type": "bind",
                "name": None,
                "destination": "/config",
                "driver": None,
                "mode": None,
                "rw": False,
                "propagation": None,
            },
            {
                "type": "volume",
                "name": "data",
                "destination": "/data",
                "driver": "local",
                "mode": "rw",
                "rw": True,
                "propagation": "",
            },
        ],
    }
    assert len(captured.docker.image_file_sha256) == 4
    assert len(captured.artifacts) == 4
    for name, artifact in captured.artifacts.items():
        assert f".{phase}." in name
        artifact_path = output / name
        assert artifact.sha256 == sha256_file(artifact_path)
        assert artifact.bytes == artifact_path.stat().st_size
        assert artifact_path.read_text() == (
            '{"postgres":"evidence"}\n'
            if name.endswith("json")
            else "name,value\nsetting,on\n"
        )
    assert len(list(output.iterdir())) == 5
    assert (
        "private-value" not in path.read_text()
        and "PRIVATE_IMAGE" not in path.read_text()
    )
    assert commands[:15] == [
        ["docker", "exec", "active-container", "/usr/local/bin/qorl-assert-config"],
        *[
            [
                "docker",
                "exec",
                "active-container",
                "/usr/local/bin/qorl-dump-postgres-state",
                mode,
            ]
            for mode in (
                "identity-json",
                "nondefaults-json",
                "all-settings-csv",
                "show-all-csv",
            )
        ],
        ["docker", "container", "inspect", "active-container"],
        ["docker", "image", "inspect", "image-id"],
        ["lscpu", "--json"],
        [
            "lscpu",
            "--json",
            "--extended=CPU,CORE,SOCKET,NODE,CACHE,ONLINE",
        ],
        [
            "findmnt",
            "--json",
            "--target",
            "/var/lib/docker",
            "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
        ],
        ["lsblk", "--json", "--nodeps", "--output", "NAME,MODEL,SERIAL,SIZE,ROTA,TYPE"],
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version",
            "--format=csv,noheader,nounits",
        ],
        ["docker", "version", "--format", "{{json .}}"],
        ["docker", "compose", "version"],
        ["docker", "info", "--format", "{{json .}}"],
    ]
    assert commands[15][:5] == ["docker", "exec", "active-container", "sh", "-c"]
    # The cgroup/shm script bytes are unchanged from the previous capture command.
    assert (
        sha256(commands[15][5].encode()).hexdigest()
        == "3f562563570d8e9fbc572622b53aebd2654d54f29d1cff9d706a2eb68969ef59"
    )
    assert commands[16:] == [
        [
            "docker",
            "exec",
            "active-container",
            "sha256sum",
            "/etc/qorl/postgres.conf",
            "/usr/share/qorl/postgres-config.expected.json",
            "/usr/share/qorl/versions.json",
            "/usr/lib/postgresql/18/lib/pg_hint_plan.so",
        ]
    ]


def test_host_capture_without_guest_frequency_limits(
    commands: list[list[str]],
    host_files: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = environment.run
    topology = {"cpus": [{"cpu": 0, "core": 0, "socket": 0, "node": 0}]}

    def command(arguments: list[str], *, check: bool = True) -> str:
        if arguments[0] == "lscpu" and arguments[-1].startswith("--extended="):
            if "MAXMHZ" in arguments[-1] or "MINMHZ" in arguments[-1]:
                # Actual util-linux 2.39 output on Lambda, where limits are unknown.
                return '{"cpus":[{"cpu":0,"maxmhz":-,"minmhz":-}]}'
            return json.dumps(topology)
        return original(arguments, check=check)

    monkeypatch.setattr(environment, "run", command)
    assert environment.capture_host().cpu.topology == topology


def test_gpu_csv_preserves_strings_and_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def command(arguments: list[str], *, check: bool = True) -> str:
        calls.append(arguments)
        if arguments[1] == "topo":
            assert check
            return "topology\n"
        assert not check
        return '0, "GPU, model", gpu-id, 0000:01:00.0, 550.1, bios\n'

    monkeypatch.setattr(environment, "run", command)
    result = environment.gpu_state()
    assert result is not None
    assert result.model_dump() == {
        "gpus": [
            {
                "index": "0",
                "name": "GPU, model",
                "uuid": "gpu-id",
                "pci_bus_id": "0000:01:00.0",
                "driver_version": "550.1",
                "vbios_version": "bios",
            }
        ],
        "topology": "topology\n",
    }
    assert calls == [
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version",
            "--format=csv,noheader,nounits",
        ],
        ["nvidia-smi", "topo", "-m"],
    ]


@pytest.mark.parametrize("stage", ["assert", "postgres", "docker"])
def test_capture_failure_propagates_and_preserves_written_artifacts(
    commands: list[list[str]],
    host_files: None,
    tmp_path: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    original = environment.run

    def command(arguments: list[str], *, check: bool = True) -> str:
        if (
            (stage == "assert" and "/usr/local/bin/qorl-assert-config" in arguments)
            or (stage == "postgres" and "identity-json" in arguments)
            or (
                stage == "docker"
                and arguments[:3] == ["docker", "container", "inspect"]
            )
        ):
            raise RuntimeError("capture failed")
        return original(arguments, check=check)

    monkeypatch.setattr(environment, "run", command)
    pool = ContainerPool("capture-test", pool_config, postgres_config)
    slot = pool.workers[0]
    slot.container_id = "active-container"
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="capture failed"):
        capture_environment(pool, slot, output, "pre")
    assert not (output / "environment.json").exists()
    assert len(list(output.glob("postgres-*"))) == (4 if stage == "docker" else 0)


def test_untrusted_json_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid(arguments: list[str], *, check: bool = True) -> str:
        return "not JSON"

    monkeypatch.setattr(environment, "run", invalid)
    with pytest.raises(ValidationError):
        environment.run_json(["lscpu", "--json"])
    with pytest.raises(ValidationError):
        environment.capture_postgres("worker", "identity-json")


@pytest.mark.parametrize(
    "optional",
    [
        None,
        {},
        {
            "RepoTags": None,
            "RepoDigests": None,
            "Config": {"Labels": None},
            "RootFS": {"Layers": None},
        },
        {
            "RepoTags": ["postgres:test"],
            "RepoDigests": ["postgres@sha256:abc"],
            "Config": {"Labels": {"version": "test"}},
            "RootFS": {"Layers": ["sha256:layer"]},
        },
    ],
)
def test_image_optional_metadata_projection(
    monkeypatch: pytest.MonkeyPatch,
    optional: JsonObject | None,
) -> None:
    raw: JsonObject = {
        "Id": "image",
        "Created": "created",
        "Os": "linux",
        "Architecture": "amd64",
        "Size": 42,
        "Config": {},
        "RootFS": {},
        "Extra": "private-value",
    }
    if optional is not None:
        raw.update(optional)

    def command(arguments: list[str], *, check: bool = True) -> str:
        assert arguments == ["docker", "image", "inspect", "image"]
        return json.dumps([raw])

    monkeypatch.setattr(environment, "run", command)
    captured = environment.safe_image_inspect("image")
    populated = optional is not None and bool(optional.get("RepoTags"))
    assert captured.repo_tags == (["postgres:test"] if populated else [])
    assert captured.repo_digests == (["postgres@sha256:abc"] if populated else [])
    assert captured.rootfs_layers == (["sha256:layer"] if populated else [])
    assert captured.labels == ({"version": "test"} if populated else {})
    assert "private-value" not in captured.model_dump_json()


def test_empty_docker_info_keeps_all_null_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def command(arguments: list[str], *, check: bool = True) -> str:
        assert arguments == ["docker", "info", "--format", "{{json .}}"]
        return "{}"

    monkeypatch.setattr(environment, "run", command)
    assert environment.safe_docker_info().model_dump(by_alias=True) == dict.fromkeys(
        [
            "ServerVersion",
            "Driver",
            "LoggingDriver",
            "CgroupDriver",
            "CgroupVersion",
            "KernelVersion",
            "OperatingSystem",
            "OSType",
            "Architecture",
            "NCPU",
            "MemTotal",
            "DockerRootDir",
            "Name",
            "LiveRestoreEnabled",
            "SecurityOptions",
        ]
    )


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("container", "Field required"),
        ("image", "Field required"),
        ("health", "Status"),
        ("invalid-json", "Invalid JSON"),
        ("list-shape", "valid list"),
    ],
)
def test_malformed_capture_errors_never_include_raw_secrets(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: str,
) -> None:
    sentinel = "PRIVATE_CAPTURE_SENTINEL"
    payload = json.dumps(
        [
            {
                "Config": {"Env": [sentinel]},
                "State": {"Health": {"Status": {"secret": sentinel}}},
            }
        ]
    )
    if kind == "invalid-json":
        payload = '{"Env":["' + sentinel + '"]'
    elif kind == "list-shape":
        payload = json.dumps({"Config": {"Env": [sentinel]}})

    def command(arguments: list[str], *, check: bool = True) -> str:
        return payload

    monkeypatch.setattr(environment, "run", command)
    with pytest.raises(ValidationError) as caught:
        if kind == "image":
            environment.safe_image_inspect("image")
        elif kind == "health":
            DockerHealth.model_validate({"Status": {"secret": sentinel}})
        else:
            environment.safe_container_inspect("worker")
    diagnostic = str(caught.value)
    assert sentinel not in diagnostic
    assert "input_value" not in diagnostic
    assert expected in diagnostic


def test_list_adapters_hide_inputs_in_nested_errors() -> None:
    for adapter in (environment.CONTAINERS, environment.IMAGES):
        with pytest.raises(ValidationError) as caught:
            adapter.validate_python(
                [
                    {
                        "Config": {"Env": ["PRIVATE_CAPTURE_SENTINEL"]},
                        "State": {
                            "Health": {"Status": {"secret": "PRIVATE_CAPTURE_SENTINEL"}}
                        },
                    }
                ]
            )
        assert "PRIVATE_CAPTURE_SENTINEL" not in str(caught.value)
        assert "Field required" in str(caught.value)


def test_malformed_health_object_cannot_become_captured_status(
    inspect_state: JsonObject,
) -> None:
    inspect_state["Health"] = "PRIVATE_CAPTURE_SENTINEL"
    with pytest.raises(ValidationError) as caught:
        safe_container_inspect("worker")
    assert "PRIVATE_CAPTURE_SENTINEL" not in str(caught.value)


def test_command_execution_preserves_working_directory_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(
        command: list[str], *, cwd: Path, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == environment.REPOSITORY_ROOT
        assert not check and capture_output and text
        return subprocess.CompletedProcess(command, 1, "stdout", "stderr")

    monkeypatch.setattr(environment.subprocess, "run", execute)
    with pytest.raises(
        RuntimeError, match=r"command failed \(1\): test-command\nstderr"
    ):
        environment.run(["test-command"])
    assert environment.run(["test-command"], check=False) == "stdout"
