from __future__ import annotations

from pathlib import Path
from typing import Any

from qorl.util import serving
from qorl.util.serving import ServedModel


class FakeProcess:
    returncode = None

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: int | None = None) -> int:
        return 0

    def kill(self) -> None:
        raise AssertionError("graceful termination should succeed")


def test_served_model_owns_process_and_log(tmp_path: Path, monkeypatch: Any) -> None:
    process = FakeProcess()
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(serving, "wait_for_server", lambda *_: None)

    with ServedModel(
        ["serve"],
        repository=tmp_path,
        log_path=tmp_path / "server.log",
        health_url="http://127.0.0.1/health",
        startup_timeout=1,
        environment={},
    ):
        assert not process.terminated

    assert process.terminated
    assert (tmp_path / "server.log").is_file()
