import json
from pathlib import Path

import pytest

from qorl.db.fixture import DatabaseFixture, FixtureError, archive_data_identity
from qorl.util.hashing import sha256_file
from qorl.workload.taskset import TaskSet


@pytest.mark.parametrize("fixture_id", ["imdb", "another-fixture"])
def test_archive_identity_without_archive_bytes(
    repository_root: Path, tmp_path: Path, fixture_id: str
) -> None:
    manifest = tmp_path / "archive.json"
    manifest.write_bytes((repository_root / "imdb/archive.json").read_bytes())
    if fixture_id != "imdb":
        with pytest.raises(FixtureError, match="fixture and workload do not match"):
            archive_data_identity(manifest, fixture_id)
    else:
        assert (
            archive_data_identity(manifest, fixture_id)
            == TaskSet.load(repository_root, "job").data_identity
        )
    assert not (tmp_path / "imdb.tar.gz").exists()


def test_archive_metadata_matches_both_workloads(repository_root: Path) -> None:
    archive = json.loads((repository_root / "imdb/archive.json").read_text())
    assert archive["fixture_id"] == "imdb"
    assert "snapshot_id" not in archive
    assert archive["source_manifest"]["filename"] == "build.json"
    assert archive["source_manifest"]["sha256"] == sha256_file(
        repository_root / "imdb/build.json"
    )
    fixture = DatabaseFixture(
        repository_root,
        repository_root / "imdb/archive.json",
        archive,
        repository_root / "imdb/imdb.tar.gz",
    )
    assert set(fixture.data_identity) == {
        "fixture_id",
        "archive_sha256",
        "postgres_system_identifier",
    }
    for workload in ("job", "ceb"):
        tasks = TaskSet.load(repository_root, workload, fixture.data_identity)
        assert tasks.inventory["inventory_id"] == workload
        assert tasks.inventory["database"] == fixture.data_identity


def test_fixture_load_checks_archive_path_size_and_checksum(tmp_path: Path) -> None:
    directory = tmp_path / "imdb"
    directory.mkdir()
    archive = directory / "imdb.tar.gz"
    archive.write_bytes(b"fixture archive")
    manifest = {
        "fixture_id": "imdb",
        "archive": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
    }
    manifest_path = directory / "archive.json"
    manifest_path.write_text(json.dumps(manifest))
    loaded = DatabaseFixture.load(tmp_path)
    loaded.verify_archive()
    archive.write_bytes(b"changed archive")
    with pytest.raises(FixtureError, match=r"(size|checksum) is incorrect"):
        loaded.verify_archive()
    manifest["archive"]["filename"] = "../imdb.tar.gz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FixtureError, match="not a basename"):
        DatabaseFixture.load(tmp_path)


class TestDatabaseFixture:
    def test_fixture_splits_data_and_runtime_identity(
        self, repository_root: Path
    ) -> None:
        fixture = DatabaseFixture(
            repository=repository_root,
            manifest_path=repository_root / "manifest.json",
            archive_path=repository_root / "manifest.tar.gz",
            manifest={
                "fixture_id": "imdb",
                "archive": {"sha256": "archive"},
                "postgresql": {"system_identifier": "system"},
                "image": {
                    "id": "sha256:image",
                },
            },
        )

        assert fixture.data_identity == {
            "fixture_id": "imdb",
            "archive_sha256": "archive",
            "postgres_system_identifier": "system",
        }
        assert fixture.runtime_identity == {
            "postgres_image_id": "sha256:image",
            "postgres_config_id": "000-pgconf-default",
        }
