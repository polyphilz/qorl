from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qorl.db.schemas import WorkerPoolConfig
from qorl.util.hashing import sha256_file

DEFAULT_POOL_CONFIG = Path("docker/worker_pool/configs/002-poolconf-4x8")
NO_SWAP_BYTES = 0
MIN_SIZE_TEXT_LENGTH = 2


def cpu_ids(cpuset: str) -> set[int]:
    values: set[int] = set()
    for part in cpuset.split(","):
        bounds = part.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU set: {cpuset}")
        values.update(range(start, end + 1))
    if not values:
        raise ValueError("CPU set must not be empty")
    return values


def size_bytes(limit: str) -> int:
    if len(limit) < MIN_SIZE_TEXT_LENGTH or limit[-1].lower() != "g":
        raise ValueError("runtime sizes must use whole GiB, such as 8g")
    gib = int(limit[:-1])
    if gib < 1:
        raise ValueError("runtime sizes must be positive")
    return gib * 1024**3


@dataclass(frozen=True)
class WorkerResources:
    index: int
    physical_core_count: int
    cpuset: str
    cpuset_mems: str
    memory_limit: str
    memory_bytes: int
    memory_swap_bytes: int
    shm_size: str
    shm_bytes: int
    port: int

    @property
    def compose_environment(self) -> dict[str, str]:
        return {
            "QORL_POSTGRES_CPUSET": self.cpuset,
            "QORL_POSTGRES_CPUSET_MEMS": self.cpuset_mems,
            "QORL_POSTGRES_MEMORY_LIMIT": self.memory_limit,
            "QORL_POSTGRES_MEMORY_BYTES": str(self.memory_bytes),
            "QORL_POSTGRES_MEMORY_SWAP_LIMIT": self.memory_limit,
            "QORL_POSTGRES_MEMORY_SWAP_BYTES": str(self.memory_swap_bytes),
            "QORL_POSTGRES_SHM_SIZE": self.shm_size,
            "QORL_POSTGRES_SHM_BYTES": str(self.shm_bytes),
            "QORL_POSTGRES_PORT": str(self.port),
        }

    def manifest(self) -> dict[str, int | str]:
        return {
            "slot": self.index,
            "physical_core_count": self.physical_core_count,
            "cpuset": self.cpuset,
            "cpuset_mems": self.cpuset_mems,
            "memory_limit": self.memory_limit,
            "memory_bytes": self.memory_bytes,
            "memory_swap_bytes": self.memory_swap_bytes,
            "shm_size": self.shm_size,
            "shm_bytes": self.shm_bytes,
            "port": self.port,
        }


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    path: Path
    sha256: str
    workers: tuple[WorkerResources, ...]
    configuration: WorkerPoolConfig

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "worker_count": len(self.workers),
            "workers": [worker.manifest() for worker in self.workers],
        }


def load_runtime_profile(repository: Path, configured: Path) -> RuntimeProfile:
    path = configured if configured.is_absolute() else repository / configured
    if path.is_dir():
        path = path / "poolconf.json"
    path = path.resolve()
    config = WorkerPoolConfig.model_validate_json(path.read_text(encoding="utf-8"))
    memory = size_bytes(config.memory_limit)
    shm = size_bytes(config.shm_size)
    cpu_ids(config.cpuset_mems)
    resources: list[WorkerResources] = []
    for index, worker in enumerate(config.workers):
        cpu_ids(worker.cpuset)
        resources.append(
            WorkerResources(
                index=index,
                physical_core_count=worker.physical_core_count,
                cpuset=worker.cpuset,
                cpuset_mems=config.cpuset_mems,
                memory_limit=config.memory_limit,
                memory_bytes=memory,
                memory_swap_bytes=NO_SWAP_BYTES,
                shm_size=config.shm_size,
                shm_bytes=shm,
                port=worker.port,
            )
        )

    parsed = [cpu_ids(worker.cpuset) for worker in resources]
    if sum(len(item) for item in parsed) != len(
        {cpu for item in parsed for cpu in item}
    ):
        raise ValueError("PostgreSQL worker CPU sets must not overlap")
    if len({worker.port for worker in resources}) != len(resources):
        raise ValueError("PostgreSQL worker ports must be distinct")
    try:
        recorded_path = path.relative_to(repository)
    except ValueError:
        recorded_path = path
    return RuntimeProfile(
        profile_id=path.parent.name,
        path=recorded_path,
        sha256=sha256_file(path),
        workers=tuple(resources),
        configuration=config,
    )


def validate_host_topology(
    resources: tuple[WorkerResources, ...],
    cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> None:
    assigned: set[tuple[str, str]] = set()
    for item in resources:
        physical: set[tuple[str, str]] = set()
        for cpu in cpu_ids(item.cpuset):
            topology = cpu_root / f"cpu{cpu}" / "topology"
            try:
                package = (topology / "physical_package_id").read_text().strip()
                core = (topology / "core_id").read_text().strip()
            except OSError as error:
                raise RuntimeError(
                    f"cannot inspect topology for logical CPU {cpu}"
                ) from error
            physical.add((package, core))
        if len(physical) != item.physical_core_count:
            raise RuntimeError(
                f"worker slot {item.index} must resolve to "
                f"{item.physical_core_count} physical cores"
            )
        if assigned & physical:
            raise RuntimeError("PostgreSQL workers share a physical CPU core")
        assigned.update(physical)
