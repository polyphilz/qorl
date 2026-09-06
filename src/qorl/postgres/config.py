from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.schemas import (
    PostgresConfigExpected,
    PostgresConfigManifest,
    PostgresSettings,
)
from qorl.util.hashing import sha256_file

POSTGRES_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PostgresConfig:
    path: Path
    pg_conf_path: Path
    expected_path: Path
    expected: PostgresConfigExpected
    pg_conf_sha256: str
    expected_sha256: str

    @classmethod
    def load(cls, configured: Path) -> PostgresConfig:
        path = (REPOSITORY_ROOT / configured).resolve()
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
    def agent_settings(self) -> PostgresSettings:
        """Select the baseline settings exposed to the agent from this config."""
        return PostgresSettings.model_validate(
            {
                name: value
                for name, value in self.expected.settings.items()
                if name in PostgresSettings.model_fields
            }
        )

    def manifest(self) -> PostgresConfigManifest:
        try:
            displayed_path = self.path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            displayed_path = self.path
        return PostgresConfigManifest(
            id=self.config_id,
            path=str(displayed_path),
            pg_conf_sha256=self.pg_conf_sha256,
            expected_sha256=self.expected_sha256,
        )
