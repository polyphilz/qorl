#!/usr/bin/env python3
"""Capture a QORL worker environment without recording container secrets.

The orchestrator calls this automatically before and after measurements. It is
also available directly for fixture verification and development diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl.util.hashing import sha256_file


def run(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        command,
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


def run_json(command: list[str]) -> Any:
    return json.loads(run(command))


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


def cpu_power_policies() -> list[dict[str, str | None]]:
    fields = (
        "affected_cpus",
        "scaling_driver",
        "scaling_governor",
        "energy_performance_preference",
        "cpuinfo_min_freq",
        "cpuinfo_max_freq",
    )
    policies: list[dict[str, str | None]] = []
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        entry: dict[str, str | None] = {"policy": policy.name}
        for field in fields:
            entry[field] = read_text(str(policy / field))
        policies.append(entry)
    return policies


def block_schedulers() -> dict[str, str]:
    schedulers: dict[str, str] = {}
    for path in sorted(Path("/sys/block").glob("*/queue/scheduler")):
        value = read_text(str(path))
        if value is not None:
            schedulers[path.parents[1].name] = value
    return schedulers


def safe_container_inspect(container: str) -> dict[str, Any]:
    raw = run_json(["docker", "container", "inspect", container])[0]
    host = raw["HostConfig"]
    state = raw["State"]
    health = state.get("Health") or {}
    return {
        "id": raw["Id"],
        "name": raw["Name"].lstrip("/"),
        "created": raw["Created"],
        "image_id": raw["Image"],
        "image_reference": raw["Config"]["Image"],
        "labels": raw["Config"].get("Labels") or {},
        "state": {
            "status": state["Status"],
            "started_at": state["StartedAt"],
            "health": health.get("Status"),
        },
        "limits": {
            "cpuset_cpus": host["CpusetCpus"],
            "cpuset_mems": host["CpusetMems"],
            "cpu_period": host["CpuPeriod"],
            "cpu_quota": host["CpuQuota"],
            "nano_cpus": host["NanoCpus"],
            "memory": host["Memory"],
            "memory_swap": host["MemorySwap"],
            "shm_size": host["ShmSize"],
            "pids_limit": host["PidsLimit"],
            "readonly_rootfs": host["ReadonlyRootfs"],
            "cgroupns_mode": host["CgroupnsMode"],
        },
        "mounts": [
            {
                "type": mount["Type"],
                "name": mount.get("Name"),
                "destination": mount["Destination"],
                "driver": mount.get("Driver"),
                "mode": mount.get("Mode"),
                "rw": mount["RW"],
                "propagation": mount["Propagation"],
            }
            for mount in raw["Mounts"]
        ],
    }


def safe_image_inspect(image_id: str) -> dict[str, Any]:
    raw = run_json(["docker", "image", "inspect", image_id])[0]
    return {
        "id": raw["Id"],
        "repo_tags": raw.get("RepoTags") or [],
        "repo_digests": raw.get("RepoDigests") or [],
        "created": raw["Created"],
        "os": raw["Os"],
        "architecture": raw["Architecture"],
        "size": raw["Size"],
        "labels": raw["Config"].get("Labels") or {},
        "rootfs_layers": raw["RootFS"].get("Layers") or [],
    }


def safe_docker_info() -> dict[str, Any]:
    raw = run_json(["docker", "info", "--format", "{{json .}}"])
    fields = (
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
    )
    return {field: raw.get(field) for field in fields}


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def container_runtime_state(container: str) -> dict[str, str]:
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
    return parse_key_values(run(["docker", "exec", container, "sh", "-c", script]))


def image_file_digests(container: str) -> dict[str, str]:
    paths = (
        "/etc/qorl/benchmark-v2.conf",
        "/usr/share/qorl/benchmark-v2.expected.json",
        "/usr/share/qorl/versions.json",
        "/usr/lib/postgresql/18/lib/pg_hint_plan.so",
    )
    output = run(["docker", "exec", container, "sha256sum", *paths])
    return {
        path: digest
        for digest, path in (line.split(maxsplit=1) for line in output.splitlines())
    }


def gpu_state() -> dict[str, Any] | None:
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
    return {
        "gpus": [
            {
                "index": row[0],
                "name": row[1],
                "uuid": row[2],
                "pci_bus_id": row[3],
                "driver_version": row[4],
                "vbios_version": row[5],
            }
            for row in rows
        ],
        "topology": run(["nvidia-smi", "topo", "-m"]),
    }


def capture_postgres(container: str, mode: str) -> str:
    return run(
        [
            "docker",
            "exec",
            container,
            "/usr/local/bin/qorl-dump-postgres-state",
            mode,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True, help="container name or ID")
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("pre", "post"))
    args = parser.parse_args()

    container = args.container
    profile_path = args.runtime_profile.resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or not isinstance(
        profile.get("profile_id"), str
    ):
        raise RuntimeError("invalid PostgreSQL runtime profile")
    try:
        displayed_profile_path = str(profile_path.relative_to(Path.cwd()))
    except ValueError:
        displayed_profile_path = str(profile_path)
    assertion_output = run(
        ["docker", "exec", container, "/usr/local/bin/qorl-assert-benchmark-config"]
    ).strip()

    artifact_contents = {
        f"postgres-identity.{args.phase}.json": capture_postgres(
            container, "identity-json"
        ),
        f"postgres-nondefaults.{args.phase}.json": capture_postgres(
            container, "nondefaults-json"
        ),
        f"postgres-settings.{args.phase}.csv": capture_postgres(
            container, "all-settings-csv"
        ),
        f"postgres-show-all.{args.phase}.csv": capture_postgres(
            container, "show-all-csv"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    for name, content in artifact_contents.items():
        path = args.output_dir / name
        write_atomic(path, content)
        artifact_paths[name] = path

    container_info = safe_container_inspect(container)
    image_info = safe_image_inspect(container_info["image_id"])

    environment = {
        "schema_version": 1,
        "benchmark_config_id": image_info["labels"].get("io.qorl.benchmark.config-id"),
        "phase": args.phase,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "assertion_output": assertion_output,
        "runtime_profile": {
            "id": profile["profile_id"],
            "path": displayed_profile_path,
            "sha256": sha256_file(profile_path),
            "configuration": profile,
        },
        "host": {
            "hostname": platform.node(),
            "os_release": parse_os_release(),
            "kernel": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "command_line": read_text("/proc/cmdline"),
            },
            "machine": {
                "sys_vendor": read_text("/sys/devices/virtual/dmi/id/sys_vendor"),
                "product_name": read_text("/sys/devices/virtual/dmi/id/product_name"),
                "product_version": read_text(
                    "/sys/devices/virtual/dmi/id/product_version"
                ),
                "bios_version": read_text("/sys/devices/virtual/dmi/id/bios_version"),
            },
            "cpu": {
                "summary": run_json(["lscpu", "--json"]),
                "topology": run_json(
                    [
                        "lscpu",
                        "--json",
                        "--extended=CPU,CORE,SOCKET,NODE,CACHE,ONLINE,MAXMHZ,MINMHZ",
                    ]
                ),
                "microcode": parse_cpu_microcode(),
                "isolated": read_text("/sys/devices/system/cpu/isolated"),
                "power_policies": cpu_power_policies(),
            },
            "memory": {
                "meminfo": parse_meminfo(),
                "transparent_huge_pages": read_text(
                    "/sys/kernel/mm/transparent_hugepage/enabled"
                ),
            },
            "storage": {
                "docker_root_mount": run_json(
                    [
                        "findmnt",
                        "--json",
                        "--target",
                        "/var/lib/docker",
                        "--output",
                        "TARGET,SOURCE,FSTYPE,OPTIONS",
                    ]
                ),
                "block_devices": run_json(
                    [
                        "lsblk",
                        "--json",
                        "--nodeps",
                        "--output",
                        "NAME,MODEL,SERIAL,SIZE,ROTA,TYPE",
                    ]
                ),
                "schedulers": block_schedulers(),
            },
            "gpus": gpu_state(),
        },
        "docker": {
            "version": run_json(["docker", "version", "--format", "{{json .}}"]),
            "compose_version": run(["docker", "compose", "version"]).strip(),
            "info": safe_docker_info(),
            "container": container_info,
            "container_runtime": container_runtime_state(container),
            "image": image_info,
            "image_file_sha256": image_file_digests(container),
        },
        "artifacts": {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(artifact_paths.items())
        },
    }

    environment_name = (
        "environment.json" if args.phase == "pre" else "environment.post.json"
    )
    environment_path = args.output_dir / environment_name
    write_atomic(
        environment_path, json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    print(f"captured {args.phase} benchmark environment in {args.output_dir}")


if __name__ == "__main__":
    main()
