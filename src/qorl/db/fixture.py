from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qorl.db.config import PostgresConfig

IMDB_ARCHIVE = Path("data/imdb.tar.gz")


class FixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabaseFixture:
    """The prepared IMDb archive restored for database workers."""

    repository: Path
    archive_path: Path

    @classmethod
    def load(cls, repository: Path) -> DatabaseFixture:
        repository = repository.resolve()
        archive = repository / IMDB_ARCHIVE
        if not archive.is_file():
            raise FixtureError(f"database archive is missing: {archive}")
        return cls(repository, archive)

    @property
    def data_identity(self) -> dict[str, str]:
        return {"fixture_id": "imdb"}

    @property
    def runtime_identity(self) -> dict[str, str]:
        return self.runtime_identity_for(PostgresConfig.load(self.repository))

    def runtime_identity_for(self, postgres_config: PostgresConfig) -> dict[str, str]:
        return postgres_config.runtime_identity().model_dump(exclude_none=True)
