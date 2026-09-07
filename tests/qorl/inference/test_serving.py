from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from qorl.inference import serving
from qorl.inference.serving import ServedModel


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


def test_served_model_owns_process_and_log(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
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


def test_failed_spawn_closes_the_log(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    log = StringIO()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: log)
    monkeypatch.setattr(
        serving.subprocess, "Popen", Mock(side_effect=OSError("spawn failed"))
    )
    with (
        pytest.raises(OSError, match="spawn failed"),
        ServedModel(
            ["serve"],
            repository=tmp_path,
            log_path=tmp_path / "server.log",
            health_url="http://127.0.0.1/health",
            startup_timeout=1,
            environment={},
        ),
    ):
        pytest.fail("failed spawn must not enter the serving scope")
    assert log.closed


def test_failed_startup_terminates_the_server(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    process = FakeProcess()
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        serving, "wait_for_server", Mock(side_effect=RuntimeError("startup failed"))
    )
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        ServedModel(
            ["serve"],
            repository=tmp_path,
            log_path=tmp_path / "server.log",
            health_url="http://127.0.0.1/health",
            startup_timeout=1,
            environment={},
        ),
    ):
        pytest.fail("failed startup must not enter the serving scope")
    assert process.terminated
