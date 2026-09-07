from pathlib import Path

import pytest

from qorl.experiment.run import main


def test_unimplemented_execution_is_not_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(tmp_path) == 1
    assert "not implemented (036 Phase 3)" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
