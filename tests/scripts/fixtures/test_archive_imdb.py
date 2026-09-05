import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl.util.hashing import sha256_file
from scripts.fixtures import archive_imdb


def test_archive_records_relative_evidence_paths_without_snapshot_id(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    build = tmp_path / "build.json"
    build.write_bytes((repository_root / "imdb/build.json").read_bytes())
    config = json.loads(build.read_text())
    evidence = tmp_path / "verification"
    evidence.mkdir()
    report = evidence / "build.json"
    report.write_text(
        json.dumps(
            {
                "phase": "build",
                "source_manifest_sha256": sha256_file(build),
                "database": {
                    "identity": {
                        "system_identifier": "system",
                        "server_version_num": "180006",
                    }
                },
            }
        )
    )
    environment = evidence / "environment.json"
    environment.write_text("{}")
    container = {
        "State": {"Running": False, "ExitCode": 0},
        "Config": {
            "Image": config["database"]["image_reference"],
            "Env": ["PGDATA=/var/lib/postgresql/18/docker"],
        },
        "Image": "sha256:image",
        "Mounts": [
            {
                "Destination": "/var/lib/postgresql",
                "Type": "volume",
                "Name": "fixture",
            }
        ],
    }
    monkeypatch.setattr(archive_imdb, "inspect_container", Mock(return_value=container))
    monkeypatch.setattr(
        archive_imdb,
        "inspect_image",
        Mock(
            return_value={
                "Architecture": "amd64",
                "Os": "linux",
            }
        ),
    )

    def archive_volume(command: list[str]) -> str:
        assert command[0:2] == ["docker", "run"]
        assert "--network=none" in command
        (tmp_path / ".imdb.tar.gz.part").write_bytes(b"prepared database")
        return ""

    monkeypatch.setattr(archive_imdb, "run", archive_volume)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "archive_imdb",
            "--container",
            "fixture",
            "--manifest",
            str(build),
            "--build-verification",
            str(report),
            "--environment-capture",
            str(environment),
            "--output-dir",
            str(tmp_path),
        ],
    )

    archive_imdb.main()

    result = json.loads((tmp_path / "archive.json").read_text())
    assert result["fixture_id"] == "imdb"
    assert "snapshot_id" not in result
    assert result["source_manifest"]["filename"] == "build.json"
    assert result["build_verification"]["filename"] == "verification/build.json"
    assert result["environment_capture"]["filename"] == "verification/environment.json"
    assert result["archive"]["filename"] == "imdb.tar.gz"
    assert result["archive"]["sha256"] == sha256_file(tmp_path / "imdb.tar.gz")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        archive_imdb.main()
