from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, start_pool
from qorl.util.io import write_json


@dataclass(frozen=True)
class TaskCompletion[ItemT, ResultT]:
    ordinal: int
    item: ItemT
    result: ResultT | None = None
    error: BaseException | None = None


class TaskRun:
    """Own a PostgreSQL pool, environment capture, and concurrent task loop."""

    def __init__(
        self,
        fixture: DatabaseFixture,
        project_name: str,
        output_dir: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        *,
        pool_field: str,
        environment_dir: Path | None = None,
        capture_environment: bool = True,
    ) -> None:
        self.fixture = fixture
        self.project_name = project_name
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.pool_field = pool_field
        self.environment_dir = environment_dir or output_dir
        self.capture_environment = capture_environment
        self.pool: WorkerPool | None = None

    def __enter__(self) -> TaskRun:
        self.start()
        return self

    def start(self) -> WorkerPool:
        if self.pool is not None:
            raise RuntimeError("task run is already started")
        try:
            self.pool = start_pool(self.fixture, self.project_name)
            self.manifest[self.pool_field] = self.pool.manifest()
            self.write()
            if self.capture_environment:
                self.capture("pre")
        except BaseException:
            self.close()
            raise
        return self.pool

    def __exit__(self, error_type: object, *_: object) -> None:
        try:
            if error_type is None and self.capture_environment:
                self.capture("post")
        finally:
            self.close()

    def capture(self, phase: str) -> None:
        if self.pool is None:
            raise RuntimeError("task run is not started")
        for slot in self.pool.workers:
            slot.container.capture_environment(
                self.environment_dir / f"worker-{slot.resources.index}", phase
            )

    def finish(self) -> None:
        if self.capture_environment:
            self.capture("post")
        self.close()

    def write(self) -> None:
        write_json(self.manifest_path, self.manifest)

    def map[ItemT, ResultT](
        self,
        items: Sequence[ItemT],
        function: Callable[[WorkerPool, ItemT], ResultT],
        *,
        concurrency: int | None = None,
        handled_errors: tuple[type[BaseException], ...] = (),
    ) -> Iterator[TaskCompletion[ItemT, ResultT]]:
        if self.pool is None:
            raise RuntimeError("task run is not started")
        workers = concurrency or len(self.pool.workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Future[ResultT], ItemT] = {
                executor.submit(function, self.pool, item): item for item in items
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    yield TaskCompletion(ordinal, item, result=future.result())
                except BaseException as error:
                    if not isinstance(error, handled_errors):
                        for pending in futures:
                            pending.cancel()
                        raise
                    yield TaskCompletion(ordinal, item, error=error)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None
