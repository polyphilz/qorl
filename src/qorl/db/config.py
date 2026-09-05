from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from qorl.db.schemas import (
    PostgresConfigExpected,
    PostgresConfigManifest,
    RuntimeIdentity,
)
from qorl.util.hashing import sha256_file

DEFAULT_POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")
POSTGRES_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PostgresConfig:
    repository: Path
    path: Path
    pg_conf_path: Path
    expected_path: Path
    expected: PostgresConfigExpected
    pg_conf_sha256: str
    expected_sha256: str

    @classmethod
    def load(
        cls,
        repository: Path,
        configured: Path = DEFAULT_POSTGRES_CONFIG,
    ) -> PostgresConfig:
        repository = repository.resolve()
        path = configured if configured.is_absolute() else repository / configured
        path = path.resolve()
        pg_conf_path = path / "pg.conf"
        expected_path = path / "config.expected.json"
        if not pg_conf_path.is_file() or not expected_path.is_file():
            raise ValueError(
                "PostgreSQL config must contain pg.conf and config.expected.json: "
                f"{path}"
            )
        try:
            expected = PostgresConfigExpected.model_validate_json(
                expected_path.read_text(encoding="utf-8")
            )
        except ValidationError as error:
            raise ValueError(
                f"invalid PostgreSQL config expectations: {path}"
            ) from error
        if expected.schema_version != POSTGRES_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported PostgreSQL config schema: {path}")
        if expected.postgres_config_id != path.name:
            raise ValueError(
                "PostgreSQL config ID must match its directory name: "
                f"{expected.postgres_config_id} != {path.name}"
            )
        return cls(
            repository=repository,
            path=path,
            pg_conf_path=pg_conf_path,
            expected_path=expected_path,
            expected=expected,
            pg_conf_sha256=sha256_file(pg_conf_path),
            expected_sha256=sha256_file(expected_path),
        )

    @property
    def config_id(self) -> str:
        return self.expected.postgres_config_id

    @property
    def compose_environment(self) -> dict[str, str]:
        return {
            "QORL_POSTGRES_CONFIG_FILE": str(self.pg_conf_path),
            "QORL_POSTGRES_EXPECTED_FILE": str(self.expected_path),
            "QORL_POSTGRES_ASSERT_SCRIPT": str(
                self.repository / "docker/postgres/scripts/assert-config.sh"
            ),
            "QORL_POSTGRES_DUMP_SCRIPT": str(
                self.repository / "docker/postgres/scripts/dump-postgres-state.sh"
            ),
        }

    def manifest(self) -> PostgresConfigManifest:
        try:
            displayed_path = self.path.relative_to(self.repository)
        except ValueError:
            displayed_path = self.path
        return PostgresConfigManifest(
            id=self.config_id,
            path=str(displayed_path),
            pg_conf_sha256=self.pg_conf_sha256,
            expected_sha256=self.expected_sha256,
        )

    def runtime_identity(self, postgres_image_id: str | None = None) -> RuntimeIdentity:
        return RuntimeIdentity(
            postgres_image_id=postgres_image_id,
            postgres_config_id=self.config_id,
        )
