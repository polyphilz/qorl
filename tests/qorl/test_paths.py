import importlib
from pathlib import Path

import pytest

from qorl import paths


def test_repository_root_does_not_depend_on_working_directory(
    repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = importlib.reload(paths).REPOSITORY_ROOT
    assert root == repository_root
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/qorl/paths.py").is_file()
