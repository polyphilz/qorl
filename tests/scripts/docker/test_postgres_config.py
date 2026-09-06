from __future__ import annotations

import json
from pathlib import Path

from qorl.plans.schemas import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    MAX_PARALLEL_WORKERS,
    NUMERIC_SETTINGS,
)


class TestPostgresConfig:
    def test_one_compose_file_serves_workers_and_fixture_loading(
        self, repository_root: Path
    ) -> None:
        compose = (repository_root / "compose.yaml").read_text()
        assert "${QORL_IMDB_DATA_DIR:-./data/raw/tables}:/qorl/imdb-data:ro" in compose
        assert "imdb-source" not in compose
        assert "QORL_IMDB_FIXTURE_ID" not in compose
        assert not (repository_root / "compose.fixture-build.yaml").exists()
        launcher = (repository_root / "src/qorl/worker_pool/containers.py").read_text()
        assert 'str(REPOSITORY_ROOT / "compose.yaml")' in launcher

    def test_each_config_has_only_the_three_declared_files(
        self, repository_root: Path
    ) -> None:
        root = repository_root / "docker/postgres/configs"
        expected = {"README.md", "pg.conf", "config.expected.json"}

        assert {path.name for path in root.iterdir()} == {
            "000-pgconf-default",
        }
        for config_dir in root.iterdir():
            assert {path.name for path in config_dir.iterdir()} == expected

    def test_fixture_loader_uses_one_fixed_postgres_and_pool_config(
        self, repository_root: Path
    ) -> None:
        script = (repository_root / "scripts/imdb/load_verify_archive.py").read_text(
            encoding="utf-8"
        )
        assert (
            'POSTGRES_CONFIG = Path("docker/postgres/configs/000-pgconf-default")'
            in script
        )
        assert (
            'POOL_CONFIG = Path("docker/worker_pool/configs/000-poolconf-1x32")'
            in script
        )
        assert "ContainerPool(" in script
        assert "scripts/docker" not in script

    def test_configs_define_every_prompt_visible_planner_setting(
        self, repository_root: Path
    ) -> None:
        names = set(BOOLEAN_SETTINGS) | set(INTEGER_SETTINGS) | set(NUMERIC_SETTINGS)
        for config_dir in (repository_root / "docker/postgres/configs").iterdir():
            config = json.loads(
                (config_dir / "config.expected.json").read_text(encoding="utf-8")
            )
            values = {name: config["settings"][name] for name in sorted(names)}

            assert set(values) == names
            assert all(isinstance(value, str) for value in values.values())

    def test_default_config_pins_memory_and_planner_controls(
        self, repository_root: Path
    ) -> None:
        root = repository_root / "docker/postgres/configs"
        stock = json.loads(
            (root / "000-pgconf-default/config.expected.json").read_text(
                encoding="utf-8"
            )
        )
        assert stock["postgres_config_id"] == "000-pgconf-default"
        assert stock["settings"]["shared_buffers"] == "16384"

        assert stock["settings"]["geqo"] == "off"
        assert (
            int(stock["settings"]["max_parallel_workers_per_gather"])
            == MAX_PARALLEL_WORKERS
        )

        def pg_settings(path: Path) -> dict[str, str]:
            return {
                name.strip(): value.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
                for name, value in (line.split("=", 1),)
            }

        stock_pg = pg_settings(root / "000-pgconf-default/pg.conf")
        assert stock_pg["qorl.postgres_config_id"] == "'000-pgconf-default'"
        assert stock_pg["shared_buffers"] == "'128MB'"
        assert stock_pg["geqo"] == "off"
        assert int(stock_pg["max_parallel_workers_per_gather"]) == MAX_PARALLEL_WORKERS
