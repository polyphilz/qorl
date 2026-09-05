from pathlib import Path

from qorl.db.fixture import DatabaseFixture


class TestDatabaseFixture:
    def test_fixture_splits_data_and_runtime_identity(
        self, repository_root: Path
    ) -> None:
        fixture = DatabaseFixture(
            repository=repository_root,
            snapshot_manifest_path=repository_root / "snapshot.json",
            archive_path=repository_root / "snapshot.tar.gz",
            snapshot={
                "fixture_id": "job-v1",
                "snapshot_id": "snapshot",
                "archive": {"sha256": "archive"},
                "postgresql": {"system_identifier": "system"},
                "image": {
                    "id": "sha256:image",
                },
            },
        )

        assert fixture.data_identity == {
            "fixture_id": "job-v1",
            "snapshot_id": "snapshot",
            "snapshot_archive_sha256": "archive",
            "postgres_system_identifier": "system",
        }
        assert fixture.runtime_identity == {
            "postgres_image_id": "sha256:image",
            "postgres_config_id": "000-pgconf-default",
        }
