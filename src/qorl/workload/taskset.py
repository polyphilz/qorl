from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qorl.db.fixture import FixtureError, data_identity

TASK_SET_PATHS = {
    "job-v1": Path("data/job/tasks.json"),
    "ceb-v1": Path("data/ceb/tasks.json"),
}


@dataclass(frozen=True)
class TaskSet:
    """A versioned collection of SQL tasks bound to a data identity."""

    repository: Path
    task_set_id: str
    inventory_path: Path
    inventory: dict[str, Any]

    @property
    def data_identity(self) -> dict[str, str]:
        return data_identity(self.inventory["database"])

    @classmethod
    def load(
        cls,
        repository: Path,
        task_set_id: str,
        expected_data_identity: dict[str, str] | None = None,
    ) -> TaskSet:
        repository = repository.resolve()
        try:
            relative_path = TASK_SET_PATHS[task_set_id]
        except KeyError as error:
            raise FixtureError(f"unknown task set: {task_set_id}") from error
        inventory_path = repository / relative_path
        if not inventory_path.is_file():
            raise FixtureError(f"required task inventory is missing: {inventory_path}")

        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        tasks = inventory.get("tasks")
        if not isinstance(tasks, list) or inventory.get("task_count") != len(tasks):
            raise FixtureError("task inventory count is incorrect")
        task_ids = [task.get("task_id") for task in tasks]
        if any(not isinstance(task_id, str) for task_id in task_ids):
            raise FixtureError("task inventory contains an invalid task ID")
        if len(task_ids) != len(set(task_ids)):
            raise FixtureError("task inventory contains duplicate task IDs")
        inventory_data_identity = data_identity(inventory.get("database", {}))
        if (
            expected_data_identity is not None
            and inventory_data_identity != data_identity(expected_data_identity)
        ):
            raise FixtureError(f"{task_set_id} requires a different database snapshot")

        return cls(
            repository=repository,
            task_set_id=task_set_id,
            inventory_path=inventory_path,
            inventory=inventory,
        )

    def load_sql(self, task: dict[str, Any]) -> str:
        relative_path = PurePosixPath(task["sql_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FixtureError(f"invalid query path: {task['task_id']}")
        path = self.inventory_path.parent.joinpath(*relative_path.parts)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != task["sql_sha256"]:
            raise FixtureError(f"query checksum mismatch: {task['task_id']}")
        return content.decode("utf-8")
