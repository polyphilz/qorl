from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.job.update_snapshot_runtime import refresh_snapshot_runtime


class TestSnapshotRuntime:
    def test_refresh_changes_only_the_image_block(self, tmp_path: Path) -> None:
        document = {
            "fixture_id": "job-v1",
            "snapshot_id": "snapshot",
            "archive": {"sha256": "archive"},
            "postgresql": {"system_identifier": "system"},
            "image": {
                "reference": "qorl-postgres:test",
                "id": "sha256:old",
                "architecture": "amd64",
                "os": "linux",
                "benchmark_config_id": "benchmark-v1",
            },
        }
        image = {
            "Id": "sha256:new",
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {"Labels": {"io.qorl.benchmark.config-id": "benchmark-v2"}},
        }
        path = tmp_path / "snapshot.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        old, new = refresh_snapshot_runtime(path, image, "benchmark-v2")
        refreshed = json.loads(path.read_text(encoding="utf-8"))

        assert (old, new) == ("sha256:old", "sha256:new")
        assert {key: value for key, value in refreshed.items() if key != "image"} == {
            key: value for key, value in document.items() if key != "image"
        }
        assert refreshed["image"]["id"] == "sha256:new"
        assert refreshed["image"]["benchmark_config_id"] == "benchmark-v2"

    def test_refresh_rejects_an_uncontracted_image(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.json"
        path.write_text(
            json.dumps(
                {
                    "image": {
                        "reference": "qorl-postgres:test",
                        "id": "sha256:old",
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="benchmark-v2"):
            refresh_snapshot_runtime(
                path,
                {
                    "Id": "sha256:new",
                    "Architecture": "amd64",
                    "Os": "linux",
                    "Config": {"Labels": {}},
                },
                "benchmark-v2",
            )
