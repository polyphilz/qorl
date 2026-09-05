from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qorl.db.config import PostgresConfig
from qorl.util.hashing import sha256_file


class FixtureError(RuntimeError):
    pass


DATA_IDENTITY_FIELDS = (
    "fixture_id",
    "snapshot_id",
    "snapshot_archive_sha256",
    "postgres_system_identifier",
)


def data_identity(value: dict[str, Any]) -> dict[str, str]:
    try:
        return {name: str(value[name]) for name in DATA_IDENTITY_FIELDS}
    except KeyError as error:
        raise FixtureError(f"database identity is missing {error.args[0]}") from error


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
            raise FixtureError(
                f"required database snapshot is missing: {manifest_path}"
            )

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
    def data_identity(self) -> dict[str, str]:
        return {
            "fixture_id": self.snapshot["fixture_id"],
            "snapshot_id": self.snapshot["snapshot_id"],
            "snapshot_archive_sha256": self.snapshot["archive"]["sha256"],
            "postgres_system_identifier": self.snapshot["postgresql"][
                "system_identifier"
            ],
        }

    @property
    def runtime_identity(self) -> dict[str, str]:
        return self.runtime_identity_for(PostgresConfig.load(self.repository))

    def runtime_identity_for(self, postgres_config: PostgresConfig) -> dict[str, str]:
        return postgres_config.runtime_identity(
            self.snapshot["image"]["id"]
        ).model_dump()

    def verify_archive(self) -> None:
        expected_bytes = self.snapshot["archive"]["bytes"]
        if self.archive_path.stat().st_size != expected_bytes:
            raise FixtureError("database snapshot archive size is incorrect")
        if sha256_file(self.archive_path) != self.snapshot["archive"]["sha256"]:
            raise FixtureError("database snapshot archive checksum is incorrect")
