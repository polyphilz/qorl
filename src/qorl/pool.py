from __future__ import annotations

import contextlib
import json
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from qorl.fixture import DatabaseFixture, sha256_file
from qorl.worker import PostgresWorker


DEFAULT_CPUSETS = (
    "0-3,16-19",
    "4-7,20-23",
    "8-11,24-27",
    "12-15,28-31",
)
DEFAULT_MEMORY_LIMIT = "8g"
DEFAULT_PORT_BASE = 56_000
DEFAULT_POOL_CONFIG = Path("configs/training/postgres-pool-v1.json")


@dataclass(frozen=True)
class WorkerResources:
    index: int
    cpuset: str
    memory_limit: str
    memory_bytes: int
    port: int

    @property
    def compose_environment(self) -> dict[str, str]:
        return {
            "QORL_POSTGRES_CPUSET": self.cpuset,
            "QORL_POSTGRES_MEMORY_LIMIT": self.memory_limit,
            "QORL_POSTGRES_MEMORY_BYTES": str(self.memory_bytes),
            "QORL_POSTGRES_PORT": str(self.port),
        }

    def manifest(self) -> dict[str, int | str]:
        return {
            "slot": self.index,
            "cpuset": self.cpuset,
            "memory_limit": self.memory_limit,
            "memory_bytes": self.memory_bytes,
            "port": self.port,
        }


@dataclass(frozen=True)
class WorkerSlot:
    resources: WorkerResources
    worker: PostgresWorker


@dataclass
class WorkerPool:
    workers: tuple[WorkerSlot, ...]
    pool_id: str
    pool_config_sha256: str

    def __post_init__(self) -> None:
        self._available: queue.Queue[WorkerSlot] = queue.Queue()
        for slot in self.workers:
            self._available.put(slot)

    @contextlib.contextmanager
    def claim_worker(self) -> Iterator[WorkerSlot]:
        slot = self._available.get()
        try:
            yield slot
        finally:
            self._available.put(slot)

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.pool_id,
            "config_sha256": self.pool_config_sha256,
            "worker_count": len(self.workers),
            "workers": [slot.resources.manifest() for slot in self.workers],
        }

    def close(self) -> None:
        for slot in reversed(self.workers):
            slot.worker.close()


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


def memory_bytes(limit: str) -> int:
    if len(limit) < 2 or limit[-1].lower() != "g":
        raise ValueError("worker memory limit must use whole GiB, such as 8g")
    gib = int(limit[:-1])
    if gib < 1:
        raise ValueError("worker memory limit must be positive")
    return gib * 1024**3


def worker_resources(
    environment: Mapping[str, str] = os.environ,
) -> tuple[WorkerResources, ...]:
    configured = environment.get("QORL_RL_WORKER_CPUSETS")
    cpusets = tuple(configured.split(";")) if configured else DEFAULT_CPUSETS
    if len(cpusets) != 4:
        raise ValueError("QORL requires exactly four PostgreSQL worker CPU sets")
    parsed = [cpu_ids(cpuset) for cpuset in cpusets]
    if sum(len(item) for item in parsed) != len(set().union(*parsed)):
        raise ValueError("PostgreSQL worker CPU sets must not overlap")

    limit = environment.get(
        "QORL_RL_WORKER_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT
    )
    limit_bytes = memory_bytes(limit)
    port_base = int(
        environment.get("QORL_RL_WORKER_PORT_BASE", DEFAULT_PORT_BASE)
    )
    if not 1 <= port_base <= 65_532:
        raise ValueError("PostgreSQL worker port base is out of range")
    return tuple(
        WorkerResources(index, cpuset, limit, limit_bytes, port_base + index)
        for index, cpuset in enumerate(cpusets)
    )


def load_pool(
    repository: Path,
    environment: Mapping[str, str] = os.environ,
) -> tuple[str, str, tuple[WorkerResources, ...]]:
    configured = Path(
        environment.get("QORL_RL_WORKER_POOL_CONFIG", DEFAULT_POOL_CONFIG)
    )
    path = configured if configured.is_absolute() else repository / configured
    config = json.loads(path.read_text(encoding="utf-8"))
    workers = config.get("workers")
    if (
        config.get("schema_version") != 1
        or not isinstance(config.get("pool_id"), str)
        or not isinstance(workers, list)
        or config.get("worker_count") != len(workers)
    ):
        raise ValueError("invalid PostgreSQL worker-pool configuration")
    if [worker.get("slot") for worker in workers] != list(range(4)):
        raise ValueError("PostgreSQL worker-pool slots must be 0 through 3")
    if any(worker.get("physical_core_count") != 4 for worker in workers):
        raise ValueError("every PostgreSQL worker must have four physical cores")
    resources = worker_resources(
        {
            "QORL_RL_WORKER_CPUSETS": ";".join(
                worker["cpuset"] for worker in workers
            ),
            "QORL_RL_WORKER_MEMORY_LIMIT": config[
                "worker_memory_limit"
            ],
            "QORL_RL_WORKER_PORT_BASE": str(config["worker_port_base"]),
        }
    )
    return config["pool_id"], sha256_file(path), resources


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
        if len(physical) != 4:
            raise RuntimeError(
                f"worker slot {item.index} must resolve to four physical cores"
            )
        if assigned & physical:
            raise RuntimeError("PostgreSQL workers share a physical CPU core")
        assigned.update(physical)


def start_pool(
    fixture: DatabaseFixture,
    project_name: str,
    environment: Mapping[str, str] = os.environ,
) -> WorkerPool:
    pool_id, config_sha256, resources_list = load_pool(
        fixture.repository, environment
    )
    validate_host_topology(resources_list)
    slots: list[WorkerSlot] = []
    try:
        for resources in resources_list:
            worker = PostgresWorker(
                fixture,
                f"{project_name}-{resources.index}",
                environment=resources.compose_environment,
            )
            slots.append(WorkerSlot(resources, worker))
            worker.start()
    except BaseException:
        for slot in reversed(slots):
            slot.worker.close()
        raise
    return WorkerPool(tuple(slots), pool_id, config_sha256)
