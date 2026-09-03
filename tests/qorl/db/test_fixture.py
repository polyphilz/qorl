from pathlib import Path
import unittest

from qorl.db.fixture import DatabaseFixture


ROOT = Path(__file__).resolve().parents[3]


class DatabaseFixtureTest(unittest.TestCase):
    def test_fixture_splits_data_and_runtime_identity(self) -> None:
        fixture = DatabaseFixture(
            repository=ROOT,
            snapshot_manifest_path=ROOT / "snapshot.json",
            archive_path=ROOT / "snapshot.tar.gz",
            snapshot={
                "fixture_id": "job-v1",
                "snapshot_id": "snapshot",
                "archive": {"sha256": "archive"},
                "postgresql": {"system_identifier": "system"},
                "image": {
                    "id": "sha256:image",
                    "benchmark_config_id": "benchmark-v2",
                },
            },
        )

        self.assertEqual(
            fixture.data_identity,
            {
                "fixture_id": "job-v1",
                "snapshot_id": "snapshot",
                "snapshot_archive_sha256": "archive",
                "postgres_system_identifier": "system",
            },
        )
        self.assertEqual(
            fixture.runtime_identity,
            {
                "postgres_image_id": "sha256:image",
                "benchmark_config_id": "benchmark-v2",
            },
        )
