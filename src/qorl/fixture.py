from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class FixtureError(RuntimeError):
    pass


TASK_SET_PATHS = {
    "job-v1": Path("data/job/job-v1/tasks.json"),
    "ceb-v1": Path("data/ceb/ceb-v1/tasks.json"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatabaseFixture:
    """The frozen PostgreSQL database restored for every worker."""

    repository: Path
    snapshot_manifest_path: Path
    snapshot: dict[str, Any]
    archive_path: Path

    @classmethod
    def load(cls, repository: Path) -> DatabaseFixture:
        repository = repository.resolve()
        manifest_path = repository / "artifacts/job-v1/job-v1.snapshot.json"
        if not manifest_path.is_file():
            raise FixtureError(f"required database snapshot is missing: {manifest_path}")

        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_name = snapshot["archive"]["filename"]
        archive_relative = PurePosixPath(archive_name)
        if archive_relative.is_absolute() or len(archive_relative.parts) != 1:
            raise FixtureError("snapshot archive filename is not a basename")
        archive_path = manifest_path.parent / archive_name
        if not archive_path.is_file():
            raise FixtureError(f"database snapshot archive is missing: {archive_path}")

        return cls(
            repository=repository,
            snapshot_manifest_path=manifest_path,
            snapshot=snapshot,
            archive_path=archive_path,
        )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "fixture_id": self.snapshot["fixture_id"],
            "snapshot_id": self.snapshot["snapshot_id"],
            "snapshot_archive_sha256": self.snapshot["archive"]["sha256"],
            "postgres_image_id": self.snapshot["image"]["id"],
            "postgres_system_identifier": self.snapshot["postgresql"][
                "system_identifier"
            ],
        }

    def verify_archive(self) -> None:
        expected_bytes = self.snapshot["archive"]["bytes"]
        if self.archive_path.stat().st_size != expected_bytes:
            raise FixtureError("database snapshot archive size is incorrect")
        if sha256_file(self.archive_path) != self.snapshot["archive"]["sha256"]:
            raise FixtureError("database snapshot archive checksum is incorrect")


@dataclass(frozen=True)
class TaskSet:
    """A versioned collection of SQL tasks bound to a database identity."""

    repository: Path
    task_set_id: str
    inventory_path: Path
    inventory: dict[str, Any]

    @classmethod
    def load(
        cls,
        repository: Path,
        task_set_id: str,
        expected_database: dict[str, str] | None = None,
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
        if expected_database is not None and inventory.get("database") != expected_database:
            raise FixtureError(
                f"{task_set_id} requires a different database snapshot"
            )

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
