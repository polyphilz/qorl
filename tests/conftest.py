from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qorl.db.fixture import DatabaseFixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def database_fixture(repository_root: Path) -> DatabaseFixture:
    database = json.loads(
        (repository_root / "data/job/tasks.json").read_text(encoding="utf-8")
    )["database"]
    benchmark = json.loads(
        (
            repository_root / "docker/postgres/contract/benchmark.expected.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = {
        "fixture_id": database["fixture_id"],
        "snapshot_id": database["snapshot_id"],
        "archive": {"sha256": database["snapshot_archive_sha256"]},
        "postgresql": {"system_identifier": database["postgres_system_identifier"]},
        "image": {
            "id": database["postgres_image_id"],
            "benchmark_config_id": benchmark["benchmark_config_id"],
        },
    }
    return DatabaseFixture(
        repository_root,
        repository_root / "artifacts/job-v1/job-v1.snapshot.json",
        snapshot,
        repository_root / "artifacts/job-v1/job-v1.snapshot.tar.gz",
    )


@pytest.fixture
def load_experiment(repository_root: Path) -> Callable[[str], dict[str, Any]]:
    def load(relative_path: str) -> dict[str, Any]:
        return runpy.run_path(repository_root / relative_path)

    return load
