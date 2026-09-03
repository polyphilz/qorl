from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def load_experiment() -> Callable[[str], dict[str, Any]]:
    def load(relative_path: str) -> dict[str, Any]:
        return runpy.run_path(ROOT / relative_path)

    return load
