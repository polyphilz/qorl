from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from qorl.taskset.exceptions import TaskSetError
from qorl.taskset.schemas import (
    BenchmarkCatalog,
    BenchmarkManifest,
    Task,
    TaskSelection,
    TemplateMetadata,
)

TASK_SET_PATHS = {
    "job": Path("benchmarks/job/tasks.json"),
    "ceb": Path("benchmarks/ceb/tasks.json"),
}


@dataclass(frozen=True)
class TaskSet:
    """A benchmark's SQL tasks and the fixture named by its source manifest."""

    task_set_id: str
    inventory_path: Path
    fixture_id: str
    tasks: list[Task]
    tasks_metadata: dict[str, TemplateMetadata]

    @classmethod
    def load(
        cls,
        repository: Path,
        task_set_id: str,
    ) -> TaskSet:
        """Load a typed benchmark catalog and check its manifest identity."""
        repository = repository.resolve()
        try:
            relative_path = TASK_SET_PATHS[task_set_id]
        except KeyError as error:
            raise TaskSetError(f"unknown task set: {task_set_id}") from error
        inventory_path = repository / relative_path
        if not inventory_path.is_file():
            raise TaskSetError(f"required task inventory is missing: {inventory_path}")

        try:
            catalog = BenchmarkCatalog.model_validate_json(inventory_path.read_bytes())
            manifest = BenchmarkManifest.model_validate_json(
                inventory_path.with_name("manifest.json").read_bytes()
            )
        except (ValidationError, OSError) as error:
            raise TaskSetError(
                f"invalid benchmark inventory or manifest: {error}"
            ) from error
        if (
            manifest.benchmark_id != task_set_id
            or catalog.benchmark_id.value != task_set_id
        ):
            raise TaskSetError("catalog or manifest references a different benchmark")

        return cls(
            task_set_id=task_set_id,
            inventory_path=inventory_path,
            fixture_id=manifest.fixture_id,
            tasks=catalog.tasks,
            tasks_metadata=catalog.tasks_metadata,
        )

    def resolve(self, selection: TaskSelection) -> list[Task]:
        """Resolve saved IDs to catalog records without resampling or reordering."""
        if selection.benchmark_id.value != self.task_set_id:
            raise TaskSetError("selection references a different benchmark")
        tasks_by_id = {task.task_id: task for task in self.tasks}
        try:
            return [tasks_by_id[task_id] for task_id in selection.task_ids]
        except KeyError as error:
            raise TaskSetError(f"unknown selected task: {error.args[0]}") from error

    def load_sql(self, task: Task) -> str:
        """Read a task's SQL and verify it against the recorded checksum."""
        relative_path = PurePosixPath(task.sql_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise TaskSetError(f"invalid query path: {task.task_id}")
        path = self.inventory_path.parent.joinpath(*relative_path.parts)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != task.sql_sha256:
            raise TaskSetError(f"query checksum mismatch: {task.task_id}")
        return content.decode("utf-8")
