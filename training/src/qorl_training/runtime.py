from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from qorl.fixture import DatabaseFixture, TaskSet
from qorl.worker import PostgresWorker


@dataclass
class QorlRuntime:
    task_set: TaskSet
    worker: PostgresWorker
    lock: threading.Lock


_runtime: QorlRuntime | None = None


def start(repository: Path) -> QorlRuntime:
    global _runtime
    if _runtime is not None:
        raise RuntimeError("QORL runtime is already started")
    fixture = DatabaseFixture.load(repository)
    worker = PostgresWorker(fixture, f"qorl-rl-{os.getpid()}")
    worker.start()
    _runtime = QorlRuntime(
        task_set=TaskSet.load(repository, "ceb-v1", fixture.identity),
        worker=worker,
        lock=threading.Lock(),
    )
    return _runtime


def current() -> QorlRuntime:
    if _runtime is None:
        raise RuntimeError("QORL runtime has not started")
    return _runtime


def stop() -> None:
    global _runtime
    runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.worker.close()
