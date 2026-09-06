from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import pytest
from pydantic import ValidationError

from qorl.postgres.config import PostgresConfig

POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")


def test_config_path_has_no_default() -> None:
    with pytest.raises(TypeError, match="configured"):
        signature(PostgresConfig.load).bind()


def test_agent_settings_preserve_configured_values(
    postgres_config: PostgresConfig,
) -> None:
    settings = postgres_config.agent_settings
    assert settings.enable_hashjoin == "on"
    assert settings.cpu_tuple_cost == "0.01"
    assert "shared_buffers" not in settings.model_dump()
    for name, value in settings.model_dump().items():
        assert value == postgres_config.expected.settings[name]


def test_agent_settings_require_every_field(postgres_config: PostgresConfig) -> None:
    del postgres_config.expected.settings["enable_hashjoin"]
    with pytest.raises(ValidationError, match="enable_hashjoin"):
        _ = postgres_config.agent_settings


class TestPostgresConfig:
    def test_loads_config_and_records_its_inputs(
        self, repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = PostgresConfig.load(POSTGRES_CONFIG)

        assert config.path == repository_root / POSTGRES_CONFIG
        assert config.config_id == "000-pgconf-default"
        assert config.expected.settings["shared_buffers"] == "16384"
        assert config.manifest().model_dump() == {
            "id": "000-pgconf-default",
            "path": str(POSTGRES_CONFIG),
            "pg_conf_sha256": config.pg_conf_sha256,
            "expected_sha256": config.expected_sha256,
        }

    def test_rejects_identity_that_differs_from_directory(self, tmp_path: Path) -> None:
        source = PostgresConfig.load(POSTGRES_CONFIG)
        (tmp_path / "pg.conf").write_text(
            source.pg_conf_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        expected = source.expected.model_dump()
        expected["postgres_config_id"] = "different"
        (tmp_path / "config.expected.json").write_text(
            json.dumps(expected), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="must match its directory name"):
            PostgresConfig.load(tmp_path)

    def test_loads_absolute_config_outside_repository(
        self, tmp_path: Path, postgres_config: PostgresConfig
    ) -> None:
        path = tmp_path / postgres_config.config_id
        path.mkdir()
        (path / "pg.conf").write_bytes(postgres_config.pg_conf_path.read_bytes())
        (path / "config.expected.json").write_bytes(
            postgres_config.expected_path.read_bytes()
        )

        config = PostgresConfig.load(path)

        assert config.path == path.resolve()
        assert config.expected == postgres_config.expected
        assert config.manifest().model_dump() == {
            **postgres_config.manifest().model_dump(),
            "path": str(path.resolve()),
        }
