from __future__ import annotations

import json
from pathlib import Path

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
            },
        }
        image = {
            "Id": "sha256:new",
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {"Labels": {}},
        }
        path = tmp_path / "snapshot.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        old, new = refresh_snapshot_runtime(path, image)
        refreshed = json.loads(path.read_text(encoding="utf-8"))

        assert (old, new) == ("sha256:old", "sha256:new")
        assert {key: value for key, value in refreshed.items() if key != "image"} == {
            key: value for key, value in document.items() if key != "image"
        }
        assert refreshed["image"]["id"] == "sha256:new"
        assert set(refreshed["image"]) == {"reference", "id", "architecture", "os"}
