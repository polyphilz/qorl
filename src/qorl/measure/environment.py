"""Gather and atomically save a worker's host, Docker and PostgreSQL evidence."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

from qorl.measure.schemas import (
    CaptureArtifact,
    CapturePhase,
    ContainerCapture,
    ContainerRuntime,
    CpuPowerPolicy,
    DockerCapture,
    DockerInfo,
    EnvironmentCapture,
    GpuCapture,
    GpuState,
    HostCapture,
    HostCpu,
    HostKernel,
    HostMachine,
    HostMemory,
    HostStorage,
    ImageCapture,
    RawCommandDocument,
    RuntimeIdentity,
    RuntimeProfile,
)
from qorl.paths import REPOSITORY_ROOT
from qorl.util.hashing import sha256_file
from qorl.util.io import display_path, write_json
from qorl.util.time import utc_now
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import WorkerSlot

RAW_COMMAND: TypeAdapter[RawCommandDocument] = TypeAdapter(
    RawCommandDocument, config=ConfigDict(hide_input_in_errors=True)
)
CONTAINERS = TypeAdapter(
    list[ContainerCapture], config=ConfigDict(hide_input_in_errors=True)
)
IMAGES = TypeAdapter(list[ImageCapture], config=ConfigDict(hide_input_in_errors=True))


def run(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def run_json(command: list[str]) -> RawCommandDocument:
    """Validate otherwise opaque, version-dependent command JSON."""
    return RAW_COMMAND.validate_json(run(command))


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_meminfo() -> dict[str, str]:
    wanted = {
        "MemTotal",
        "SwapTotal",
        "HugePages_Total",
        "HugePages_Free",
        "Hugepagesize",
    }
    values: dict[str, str] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            values[key] = value.strip()
    return values


def parse_cpu_microcode() -> str | None:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("microcode"):
            return line.split(":", 1)[1].strip()
    return None


def cpu_power_policies() -> list[CpuPowerPolicy]:
    return [
        CpuPowerPolicy(
            policy=policy.name,
            affected_cpus=read_text(str(policy / "affected_cpus")),
            scaling_driver=read_text(str(policy / "scaling_driver")),
            scaling_governor=read_text(str(policy / "scaling_governor")),
            energy_performance_preference=read_text(
                str(policy / "energy_performance_preference")
            ),
            cpuinfo_min_freq=read_text(str(policy / "cpuinfo_min_freq")),
            cpuinfo_max_freq=read_text(str(policy / "cpuinfo_max_freq")),
        )
        for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))
    ]


def block_schedulers() -> dict[str, str]:
    schedulers: dict[str, str] = {}
    for path in sorted(Path("/sys/block").glob("*/queue/scheduler")):
        value = read_text(str(path))
        if value is not None:
            schedulers[path.parents[1].name] = value
    return schedulers


def safe_container_inspect(container: str) -> ContainerCapture:
    return CONTAINERS.validate_python(
        run_json(["docker", "container", "inspect", container])
    )[0]


def safe_image_inspect(image_id: str) -> ImageCapture:
    return IMAGES.validate_python(run_json(["docker", "image", "inspect", image_id]))[0]


def safe_docker_info() -> DockerInfo:
    return DockerInfo.model_validate(
        run_json(["docker", "info", "--format", "{{json .}}"])
    )


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def container_runtime_state(container: str) -> ContainerRuntime:
    script = r"""
printf 'cpus_allowed_list='
awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status
printf 'mems_allowed_list='
awk '/^Mems_allowed_list:/ {print $2}' /proc/self/status
printf 'cpu_max='
cat /sys/fs/cgroup/cpu.max
printf 'memory_max='
cat /sys/fs/cgroup/memory.max
printf 'memory_swap_max='
cat /sys/fs/cgroup/memory.swap.max
read shm_block_size shm_blocks <<EOF
$(stat --file-system --format='%S %b' /dev/shm)
EOF
printf 'shm_size_bytes=%s\n' "$((shm_block_size * shm_blocks))"
"""
    return ContainerRuntime.model_validate(
        parse_key_values(run(["docker", "exec", container, "sh", "-c", script]))
    )


def image_file_digests(container: str) -> dict[str, str]:
    paths = (
        "/etc/qorl/postgres.conf",
        "/usr/share/qorl/postgres-config.expected.json",
        "/usr/share/qorl/versions.json",
        "/usr/lib/postgresql/18/lib/pg_hint_plan.so",
    )
    output = run(["docker", "exec", container, "sha256sum", *paths])
    return {
        path: digest
        for digest, path in (line.split(maxsplit=1) for line in output.splitlines())
    }


def gpu_state() -> GpuState | None:
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if not query.strip():
        return None
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    return GpuState(
        gpus=[
            GpuCapture(
                index=row[0],
                name=row[1],
                uuid=row[2],
                pci_bus_id=row[3],
                driver_version=row[4],
                vbios_version=row[5],
            )
            for row in rows
        ],
        topology=run(["nvidia-smi", "topo", "-m"]),
    )


def capture_postgres(container: str, mode: str) -> str:
    output = run(
        [
            "docker",
            "exec",
            container,
            "/usr/local/bin/qorl-dump-postgres-state",
            mode,
        ]
    )
    if mode in {"identity-json", "nondefaults-json"}:
        RAW_COMMAND.validate_json(output)
    return output


def capture_host() -> HostCapture:
    return HostCapture(
        hostname=platform.node(),
        os_release=parse_os_release(),
        kernel=HostKernel(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            command_line=read_text("/proc/cmdline"),
        ),
        machine=HostMachine(
            sys_vendor=read_text("/sys/devices/virtual/dmi/id/sys_vendor"),
            product_name=read_text("/sys/devices/virtual/dmi/id/product_name"),
            product_version=read_text("/sys/devices/virtual/dmi/id/product_version"),
            bios_version=read_text("/sys/devices/virtual/dmi/id/bios_version"),
        ),
        cpu=HostCpu(
            summary=run_json(["lscpu", "--json"]),
            topology=run_json(
                [
                    "lscpu",
                    "--json",
                    # Some guests expose no frequency limits; util-linux 2.39
                    # emits bare '-' values for those columns, invalidating JSON.
                    # Available frequency evidence remains in summary/power_policies.
                    "--extended=CPU,CORE,SOCKET,NODE,CACHE,ONLINE",
                ]
            ),
            microcode=parse_cpu_microcode(),
            isolated=read_text("/sys/devices/system/cpu/isolated"),
            power_policies=cpu_power_policies(),
        ),
        memory=HostMemory(
            meminfo=parse_meminfo(),
            transparent_huge_pages=read_text(
                "/sys/kernel/mm/transparent_hugepage/enabled"
            ),
        ),
        storage=HostStorage(
            docker_root_mount=run_json(
                [
                    "findmnt",
                    "--json",
                    "--target",
                    "/var/lib/docker",
                    "--output",
                    "TARGET,SOURCE,FSTYPE,OPTIONS",
                ]
            ),
            block_devices=run_json(
                [
                    "lsblk",
                    "--json",
                    "--nodeps",
                    "--output",
                    "NAME,MODEL,SERIAL,SIZE,ROTA,TYPE",
                ]
            ),
            schedulers=block_schedulers(),
        ),
        gpus=gpu_state(),
    )


def capture_environment(
    pool: ContainerPool,
    slot: WorkerSlot,
    output_dir: Path,
    phase: CapturePhase,
) -> None:
    container = slot.container_id
    profile = pool.pool_config
    postgres_config = pool.postgres_config
    assertion_output = run(
        ["docker", "exec", container, "/usr/local/bin/qorl-assert-config"]
    ).strip()
    artifact_contents = {
        f"postgres-identity.{phase}.json": capture_postgres(container, "identity-json"),
        f"postgres-nondefaults.{phase}.json": capture_postgres(
            container, "nondefaults-json"
        ),
        f"postgres-settings.{phase}.csv": capture_postgres(
            container, "all-settings-csv"
        ),
        f"postgres-show-all.{phase}.csv": capture_postgres(container, "show-all-csv"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, CaptureArtifact] = {}
    for name, content in artifact_contents.items():
        path = output_dir / name
        write_atomic(path, content)
        artifacts[name] = CaptureArtifact(
            sha256=sha256_file(path), bytes=path.stat().st_size
        )
    container_info = safe_container_inspect(container)
    image_info = safe_image_inspect(container_info.image_id)
    environment = EnvironmentCapture(
        runtime_identity=RuntimeIdentity(
            postgres_image_id=container_info.image_id,
            postgres_config_id=postgres_config.config_id,
        ),
        postgres_config=postgres_config.manifest(),
        phase=phase,
        captured_at_utc=utc_now(),
        assertion_output=assertion_output,
        runtime_profile=RuntimeProfile(
            id=profile.profile_id,
            path=display_path(
                REPOSITORY_ROOT, (REPOSITORY_ROOT / profile.path).resolve()
            ),
            sha256=profile.sha256,
            configuration=profile.configuration,
        ),
        host=capture_host(),
        docker=DockerCapture(
            version=run_json(["docker", "version", "--format", "{{json .}}"]),
            compose_version=run(["docker", "compose", "version"]).strip(),
            info=safe_docker_info(),
            container=container_info,
            container_runtime=container_runtime_state(container),
            image=image_info,
            image_file_sha256=image_file_digests(container),
        ),
        artifacts=artifacts,
    )
    environment_name = "environment.json" if phase == "pre" else "environment.post.json"
    write_json(
        output_dir / environment_name,
        environment.model_dump(mode="json", by_alias=True),
    )
