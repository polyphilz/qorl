from pathlib import Path

import pytest

from qorl.db.config import PostgresConfig
from qorl.db.fixture import DatabaseFixture, FixtureError


def test_fixture_load_needs_only_the_prepared_archive(tmp_path: Path) -> None:
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"prepared archive")

    fixture = DatabaseFixture.load(tmp_path)

    assert fixture.archive_path == archive
    assert fixture.repository == tmp_path
    assert fixture.data_identity == {"fixture_id": "imdb"}
    assert list(archive.parent.iterdir()) == [archive]


def test_missing_archive_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="database archive is missing"):
        DatabaseFixture.load(tmp_path)


def test_runtime_records_the_selected_postgres_config(
    repository_root: Path, database_fixture: DatabaseFixture
) -> None:
    config = PostgresConfig.load(
        repository_root, Path("docker/postgres/configs/001-pgconf")
    )
    assert database_fixture.runtime_identity_for(config) == {
        "postgres_config_id": "001-pgconf"
    }
