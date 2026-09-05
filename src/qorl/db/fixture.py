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
    "archive_sha256",
    "postgres_system_identifier",
)


def data_identity(value: dict[str, Any]) -> dict[str, str]:
    try:
        return {name: str(value[name]) for name in DATA_IDENTITY_FIELDS}
    except KeyError as error:
        raise FixtureError(f"database identity is missing {error.args[0]}") from error


def archive_data_identity(manifest_path: Path, fixture_id: str) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["fixture_id"] != fixture_id:
        raise FixtureError("database fixture and workload do not match")
    return {
        "fixture_id": manifest["fixture_id"],
        "archive_sha256": manifest["archive"]["sha256"],
        "postgres_system_identifier": manifest["postgresql"]["system_identifier"],
    }


@dataclass(frozen=True)
class DatabaseFixture:
    """The frozen PostgreSQL database restored for every worker."""

    repository: Path
    manifest_path: Path
    manifest: dict[str, Any]
    archive_path: Path

    @classmethod
    def load(cls, repository: Path) -> DatabaseFixture:
        repository = repository.resolve()
        manifest_path = repository / "imdb/archive.json"
        if not manifest_path.is_file():
            raise FixtureError(
                f"required database manifest is missing: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_name = manifest["archive"]["filename"]
        archive_relative = PurePosixPath(archive_name)
        if archive_relative.is_absolute() or len(archive_relative.parts) != 1:
            raise FixtureError("database archive filename is not a basename")
        archive_path = manifest_path.parent / archive_name
        if not archive_path.is_file():
            raise FixtureError(f"database archive is missing: {archive_path}")

        return cls(
            repository=repository,
            manifest_path=manifest_path,
            manifest=manifest,
            archive_path=archive_path,
        )

    @property
    def data_identity(self) -> dict[str, str]:
        return {
            "fixture_id": self.manifest["fixture_id"],
            "archive_sha256": self.manifest["archive"]["sha256"],
            "postgres_system_identifier": self.manifest["postgresql"][
                "system_identifier"
            ],
        }

    @property
    def runtime_identity(self) -> dict[str, str]:
        return self.runtime_identity_for(PostgresConfig.load(self.repository))

    def runtime_identity_for(self, postgres_config: PostgresConfig) -> dict[str, str]:
        return postgres_config.runtime_identity(
            self.manifest["image"]["id"]
        ).model_dump()

    def verify_archive(self) -> None:
        expected_bytes = self.manifest["archive"]["bytes"]
        if self.archive_path.stat().st_size != expected_bytes:
            raise FixtureError("database archive size is incorrect")
        if sha256_file(self.archive_path) != self.manifest["archive"]["sha256"]:
            raise FixtureError("database archive checksum is incorrect")
