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
    manifest = json.loads(
        (repository_root / "imdb/archive.json").read_text(encoding="utf-8")
    )
    return DatabaseFixture(
        repository_root,
        repository_root / "imdb/archive.json",
        manifest,
        repository_root / "imdb/imdb.tar.gz",
    )


@pytest.fixture
def load_experiment(repository_root: Path) -> Callable[[str], dict[str, Any]]:
    def load(relative_path: str) -> dict[str, Any]:
        return runpy.run_path(repository_root / relative_path)

    return load
