from __future__ import annotations

import json
from pathlib import Path

from scripts.fixtures.update_fixture_runtime import refresh_fixture_runtime


class TestSnapshotRuntime:
    def test_refresh_changes_only_the_image_block(self, tmp_path: Path) -> None:
        document = {
            "fixture_id": "imdb",
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
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        old, new = refresh_fixture_runtime(path, image)
        refreshed = json.loads(path.read_text(encoding="utf-8"))

        assert (old, new) == ("sha256:old", "sha256:new")
        assert {key: value for key, value in refreshed.items() if key != "image"} == {
            key: value for key, value in document.items() if key != "image"
        }
        assert refreshed["image"]["id"] == "sha256:new"
        assert set(refreshed["image"]) == {"reference", "id", "architecture", "os"}
