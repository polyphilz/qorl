from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from qorl.workload.schemas import TASKS_ADAPTER, BenchmarkManifest, Task

TASK_SET_PATHS = {
    "job": Path("benchmarks/job/tasks.json"),
    "ceb": Path("benchmarks/ceb/tasks.json"),
}


class TaskSetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskSet:
    """A benchmark's SQL tasks and the fixture named by its source manifest."""

    repository: Path
    task_set_id: str
    inventory_path: Path
    fixture_id: str
    tasks: list[Task]

    @property
    def data_identity(self) -> dict[str, str]:
        return {"fixture_id": self.fixture_id}

    @classmethod
    def load(
        cls,
        repository: Path,
        task_set_id: str,
    ) -> TaskSet:
        repository = repository.resolve()
        try:
            relative_path = TASK_SET_PATHS[task_set_id]
        except KeyError as error:
            raise TaskSetError(f"unknown task set: {task_set_id}") from error
        inventory_path = repository / relative_path
        if not inventory_path.is_file():
            raise TaskSetError(f"required task inventory is missing: {inventory_path}")

        try:
            tasks = TASKS_ADAPTER.validate_json(inventory_path.read_bytes())
            manifest = BenchmarkManifest.model_validate_json(
                inventory_path.with_name("manifest.json").read_bytes()
            )
        except (ValidationError, OSError) as error:
            raise TaskSetError(
                f"invalid benchmark inventory or manifest: {error}"
            ) from error
        if manifest.workload_id != task_set_id:
            raise TaskSetError("benchmark manifest references a different workload")
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise TaskSetError("task inventory contains duplicate task IDs")

        return cls(
            repository=repository,
            task_set_id=task_set_id,
            inventory_path=inventory_path,
            fixture_id=manifest.fixture_id,
            tasks=tasks,
        )

    def load_sql(self, task: dict[str, Any]) -> str:
        relative_path = PurePosixPath(task["sql_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise TaskSetError(f"invalid query path: {task['task_id']}")
        path = self.inventory_path.parent.joinpath(*relative_path.parts)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != task["sql_sha256"]:
            raise TaskSetError(f"query checksum mismatch: {task['task_id']}")
        return content.decode("utf-8")
