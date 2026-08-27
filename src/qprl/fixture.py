from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FixtureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JobFixture:
    repository: Path
    inventory_path: Path
    inventory: dict[str, Any]
    snapshot_manifest_path: Path
    snapshot: dict[str, Any]
    archive_path: Path

    @classmethod
    def load(cls, repository: Path) -> JobFixture:
        repository = repository.resolve()
        inventory_path = repository / "data/job/job-v1/tasks.json"
        snapshot_manifest_path = (
            repository / "artifacts/job-v1/job-v1.snapshot.json"
        )
        for path in (inventory_path, snapshot_manifest_path):
            if not path.is_file():
                raise FixtureError(f"required job-v1 file is missing: {path}")

        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        archive_name = snapshot["archive"]["filename"]
        if Path(archive_name).name != archive_name:
            raise FixtureError("snapshot archive filename is not a basename")
        archive_path = snapshot_manifest_path.parent / archive_name
        if not archive_path.is_file():
            raise FixtureError(f"job-v1 snapshot archive is missing: {archive_path}")

        database = inventory["database"]
        expected = {
            "fixture_id": snapshot["fixture_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_archive_sha256": snapshot["archive"]["sha256"],
            "postgres_image_id": snapshot["image"]["id"],
            "postgres_system_identifier": snapshot["postgresql"][
                "system_identifier"
            ],
        }
        if database != expected:
            raise FixtureError("task inventory and snapshot identities do not match")
        if inventory["task_count"] != len(inventory["tasks"]):
            raise FixtureError("task inventory count is incorrect")

        return cls(
            repository=repository,
            inventory_path=inventory_path,
            inventory=inventory,
            snapshot_manifest_path=snapshot_manifest_path,
            snapshot=snapshot,
            archive_path=archive_path,
        )

    def verify_archive(self) -> None:
        expected_bytes = self.snapshot["archive"]["bytes"]
        if self.archive_path.stat().st_size != expected_bytes:
            raise FixtureError("job-v1 snapshot archive size is incorrect")
        if sha256_file(self.archive_path) != self.snapshot["archive"]["sha256"]:
            raise FixtureError("job-v1 snapshot archive checksum is incorrect")
