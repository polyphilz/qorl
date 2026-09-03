from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qorl.db.fixture import data_identity
from qorl.util.hashing import sha256_file
from qorl.workload.taskset import TaskSet

TIMEOUT_FLOOR_MS = 5_000
TIMEOUT_MULTIPLIER = 3
GLOBAL_TIMEOUT_MS = 120_000


def task_timeout_ms(
    calibrated_default_ms: float,
    global_cap_ms: int = GLOBAL_TIMEOUT_MS,
) -> int:
    return min(
        global_cap_ms,
        max(
            TIMEOUT_FLOOR_MS,
            math.ceil(TIMEOUT_MULTIPLIER * calibrated_default_ms),
        ),
    )


@dataclass(frozen=True)
class TaskTimeout:
    task_id: str
    calibrated_default_ms: float
    timeout_ms: int
    plan_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class CalibratedTimeouts:
    path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    by_task_id: dict[str, TaskTimeout]

    @classmethod
    def load(
        cls,
        repository: Path,
        path: Path,
        task_set: TaskSet,
        expected_runtime_identity: dict[str, str] | None = None,
    ) -> CalibratedTimeouts:
        path = path if path.is_absolute() else repository / path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise RuntimeError("unsupported calibrated-timeout manifest")
        recorded_data_identity = manifest.get(
            "data_identity", manifest.get("database", {})
        )
        if data_identity(recorded_data_identity) != task_set.data_identity:
            raise RuntimeError("calibrated timeouts use a different database")
        recorded_runtime_identity = manifest.get("runtime_identity")
        if (
            expected_runtime_identity is not None
            and recorded_runtime_identity is not None
            and recorded_runtime_identity != expected_runtime_identity
        ):
            raise RuntimeError("calibrated timeouts use a different runtime")
        if (
            expected_runtime_identity is not None
            and recorded_runtime_identity is None
            and manifest.get("database", {}).get("postgres_image_id")
            != expected_runtime_identity["postgres_image_id"]
        ):
            raise RuntimeError("calibrated timeouts use a different runtime")

        selection = manifest.get("selection", {})
        selection_path = repository / selection.get("path", "")
        if not selection_path.is_file():
            raise RuntimeError("calibrated-timeout selection is missing")
        if sha256_file(selection_path) != selection.get("sha256"):
            raise RuntimeError("calibrated-timeout selection checksum differs")
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        try:
            selected_ids = [
                item["task_id"] for item in selected["splits"][selection["split"]]
            ]
        except (KeyError, TypeError) as error:
            raise RuntimeError("calibrated-timeout selection is invalid") from error

        algorithm = manifest.get("algorithm", {})
        if algorithm != {
            "global_cap_ms": GLOBAL_TIMEOUT_MS,
            "minimum_ms": TIMEOUT_FLOOR_MS,
            "multiplier": TIMEOUT_MULTIPLIER,
        }:
            raise RuntimeError("calibrated-timeout algorithm differs")

        entries = manifest.get("tasks")
        if not isinstance(entries, list) or manifest.get("task_count") != len(entries):
            raise RuntimeError("calibrated-timeout task count differs")
        by_task_id: dict[str, TaskTimeout] = {}
        for entry in entries:
            task_id = entry.get("task_id")
            median = entry.get("calibrated_default_median_ms")
            timeout = entry.get("timeout_ms")
            hashes = entry.get("plan_sha256s")
            if (
                not isinstance(task_id, str)
                or isinstance(median, bool)
                or not isinstance(median, (int, float))
                or median <= 0
                or not isinstance(timeout, int)
                or timeout != task_timeout_ms(float(median))
                or not isinstance(hashes, list)
                or not hashes
                or any(not isinstance(value, str) for value in hashes)
            ):
                raise RuntimeError(f"invalid calibrated timeout: {task_id}")
            if task_id in by_task_id:
                raise RuntimeError("calibrated-timeout task IDs are duplicated")
            by_task_id[task_id] = TaskTimeout(
                task_id,
                float(median),
                timeout,
                tuple(hashes),
            )
        if list(by_task_id) != selected_ids:
            raise RuntimeError("calibrated timeouts do not match their selection")
        return cls(path, sha256_file(path), manifest, by_task_id)

    def identity(self) -> dict[str, str]:
        return {
            "id": self.manifest["manifest_id"],
            "sha256": self.manifest_sha256,
        }

    def task(self, task_id: str) -> TaskTimeout:
        try:
            return self.by_task_id[task_id]
        except KeyError as error:
            raise RuntimeError(f"task has no calibrated timeout: {task_id}") from error
