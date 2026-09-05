from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qorl.db.resources import (
    DEFAULT_POOL_CONFIG,
    cpu_ids,
    load_runtime_profile,
    validate_host_topology,
)
from qorl.util.hashing import sha256_file


@pytest.fixture
def cpu_topology(tmp_path: Path) -> Path:
    for cpu in range(32):
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0")
        (topology / "core_id").write_text(str(cpu % 16))
    return tmp_path


@pytest.mark.parametrize(
    ("config_id", "worker_count", "memory_gib", "physical_cores", "ports"),
    [
        ("000-poolconf-1x32", 1, 32, 16, [55432]),
        ("001-poolconf-2x16", 2, 16, 8, [56000, 56001]),
        ("002-poolconf-4x8", 4, 8, 4, [56000, 56001, 56002, 56003]),
    ],
)
def test_configs_preserve_total_resources_and_resolve_each_worker(
    repository_root: Path,
    cpu_topology: Path,
    config_id: str,
    worker_count: int,
    memory_gib: int,
    physical_cores: int,
    ports: list[int],
) -> None:
    config_dir = Path("docker/worker_pool/configs") / config_id
    profile = load_runtime_profile(repository_root, config_dir)
    assert profile == load_runtime_profile(
        repository_root, config_dir / "poolconf.json"
    )
    assert profile.profile_id == config_id
    assert profile.path == config_dir / "poolconf.json"
    assert profile.sha256 == sha256_file(repository_root / profile.path)
    assert len(profile.workers) == worker_count
    assert [worker.port for worker in profile.workers] == ports
    assert sum(worker.memory_bytes for worker in profile.workers) == 32 * 1024**3
    assert sum(worker.physical_core_count for worker in profile.workers) == 16
    assert set().union(*(cpu_ids(worker.cpuset) for worker in profile.workers)) == set(
        range(32)
    )
    validate_host_topology(profile.workers, cpu_topology)

    for index, worker in enumerate(profile.workers):
        assert worker.index == index
        assert worker.physical_core_count == physical_cores
        assert worker.compose_environment == {
            "QORL_POSTGRES_CPUSET": worker.cpuset,
            "QORL_POSTGRES_CPUSET_MEMS": "0",
            "QORL_POSTGRES_MEMORY_LIMIT": f"{memory_gib}g",
            "QORL_POSTGRES_MEMORY_BYTES": str(memory_gib * 1024**3),
            "QORL_POSTGRES_MEMORY_SWAP_LIMIT": f"{memory_gib}g",
            "QORL_POSTGRES_MEMORY_SWAP_BYTES": "0",
            "QORL_POSTGRES_SHM_SIZE": "1g",
            "QORL_POSTGRES_SHM_BYTES": str(1024**3),
            "QORL_POSTGRES_PORT": str(ports[index]),
        }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", [], "at least 1 item"),
        ("memory_limit", "0g", "must be positive"),
        ("shm_size", "-1g", "must be positive"),
        ("cpuset_mems", "invalid", "invalid literal"),
        ("worker_count", 4, "Extra inputs"),
    ],
)
def test_rejects_invalid_pool_settings(
    repository_root: Path, tmp_path: Path, field: str, value: object, message: str
) -> None:
    raw = json.loads(
        (repository_root / DEFAULT_POOL_CONFIG / "poolconf.json").read_text()
    )
    raw[field] = value
    (tmp_path / "poolconf.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_runtime_profile(repository_root, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cpuset", "0-3,16-19", "must not overlap"),
        ("port", 56000, "ports must be distinct"),
        ("port", 65536, "less than or equal to"),
        ("port", True, "valid integer"),
        ("physical_core_count", 0, "greater than or equal to"),
    ],
)
def test_rejects_invalid_worker_settings(
    repository_root: Path, tmp_path: Path, field: str, value: object, message: str
) -> None:
    raw = json.loads(
        (repository_root / DEFAULT_POOL_CONFIG / "poolconf.json").read_text()
    )
    raw["workers"][1][field] = value
    (tmp_path / "poolconf.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_runtime_profile(repository_root, tmp_path)


def test_topology_rejects_incorrect_core_counts_and_shared_siblings(
    repository_root: Path, cpu_topology: Path
) -> None:
    profile = load_runtime_profile(repository_root, DEFAULT_POOL_CONFIG)
    with pytest.raises(RuntimeError, match="physical cores"):
        validate_host_topology(
            (replace(profile.workers[0], cpuset="0-1"),), cpu_topology
        )
    with pytest.raises(RuntimeError, match="share a physical CPU core"):
        validate_host_topology(
            (
                replace(profile.workers[0], cpuset="0-3"),
                replace(profile.workers[1], cpuset="16-19"),
            ),
            cpu_topology,
        )
