from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.db.config import DEFAULT_POSTGRES_CONFIG, PostgresConfig


class TestPostgresConfig:
    def test_loads_config_and_records_its_inputs(self, repository_root: Path) -> None:
        config = PostgresConfig.load(repository_root)

        assert config.path == repository_root / DEFAULT_POSTGRES_CONFIG
        assert config.config_id == "000-pgconf-default"
        assert config.expected.settings["shared_buffers"] == "16384"
        assert config.manifest().model_dump() == {
            "id": "000-pgconf-default",
            "path": str(DEFAULT_POSTGRES_CONFIG),
            "pg_conf_sha256": config.pg_conf_sha256,
            "expected_sha256": config.expected_sha256,
        }
        assert config.runtime_identity("sha256:image").model_dump() == {
            "postgres_image_id": "sha256:image",
            "postgres_config_id": "000-pgconf-default",
        }

    def test_loads_one_gib_variant(self, repository_root: Path) -> None:
        config = PostgresConfig.load(
            repository_root, Path("docker/postgres/configs/001-pgconf")
        )

        assert config.config_id == "001-pgconf"
        assert config.expected.settings["shared_buffers"] == "131072"

    def test_rejects_identity_that_differs_from_directory(
        self, repository_root: Path, tmp_path: Path
    ) -> None:
        source = PostgresConfig.load(repository_root)
        (tmp_path / "pg.conf").write_text(
            source.pg_conf_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        expected = source.expected.model_dump()
        expected["postgres_config_id"] = "different"
        (tmp_path / "config.expected.json").write_text(
            json.dumps(expected), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="must match its directory name"):
            PostgresConfig.load(repository_root, tmp_path)
