from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO


def wait_for_server(url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=10):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise RuntimeError(f"vLLM did not become ready within {timeout} seconds")


class ServedModel:
    """Own a model server process for exactly one evaluation scope."""

    def __init__(
        self,
        command: list[str],
        *,
        repository: Path,
        log_path: Path,
        health_url: str,
        startup_timeout: int,
        environment: Mapping[str, str],
    ) -> None:
        self.command = command
        self.repository = repository
        self.log_path = log_path
        self.health_url = health_url
        self.startup_timeout = startup_timeout
        self.environment = environment
        self._log: TextIO | None = None
        self.process: subprocess.Popen[Any] | None = None

    def __enter__(self) -> ServedModel:
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.repository,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=self.environment,
        )
        try:
            wait_for_server(self.health_url, self.process, self.startup_timeout)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None
