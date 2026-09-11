"""Lease PostgreSQL workers for complete measurement phases within an RL episode."""

import random
import time
from collections.abc import Generator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from threading import Event

from pydantic import JsonValue

from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import Baseline, Candidate, Outcome
from qorl.postgres.client import PostgresClient
from qorl.postgres.schemas import ExplainResult, WorkerAllocation
from qorl.rl.runtime import QorlRuntime
from qorl.rl.schemas import DatabaseWorkerLease
from qorl.worker_pool.schemas import WorkerSlot


class PooledExecutor:
    """One episode's SQL executor; only synchronous database work holds a worker."""

    def __init__(self, pool: QorlRuntime, cancel: Event) -> None:
        capacities = {
            (
                slot.resources.physical_core_count,
                slot.resources.memory_bytes,
                slot.resources.memory_swap_bytes,
                slot.resources.shm_bytes,
            )
            for slot in pool.workers
        }
        if len(capacities) != 1:
            raise ValueError(
                "measurement leases require uniform worker resource limits"
            )
        self.pool = pool
        self.cancel = cancel
        self.settings = pool.settings
        self.indexes = pool.indexes
        self.allocation: WorkerAllocation | None = None
        self.leases: list[DatabaseWorkerLease] = []
        self._slot: WorkerSlot | None = None

    @contextmanager
    def hold(self, operation: str) -> Generator[PostgresClient, None, None]:
        if self.cancel.is_set():
            raise CancelledError("rollout cancelled")
        if self._slot is not None:
            yield self._slot.client
            return
        waiting = time.monotonic()
        with self.pool.claim_worker(cancel=self.cancel) as slot:
            acquired = time.monotonic()
            self._slot = slot
            # The initial observation describes the baseline worker's verified limits.
            # Every later worker has the same resource capacities.
            if not self.leases:
                self.allocation = slot.client.allocation
            try:
                yield slot.client
            finally:
                self._slot = None
                self.leases.append(
                    DatabaseWorkerLease(
                        operation=operation,
                        worker_slot=slot.resources.index,
                        wait_seconds=acquired - waiting,
                        held_seconds=time.monotonic() - acquired,
                    )
                )

    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        with self.hold("query") as client:
            return client.explain(sql, timeout_ms, analyze=analyze, hint=hint)

    def admin_sql(self, sql: str) -> str:
        with self.hold("inspection") as client:
            return client.admin_sql(sql)


class PooledRolloutEvaluator(RolloutEvaluator[PooledExecutor]):
    """Keep warmups, measurements and final pairs indivisible on one worker."""

    def start(self) -> Baseline:
        with self.worker.hold("initial_default"):
            return super().start()

    def evaluate(self, raw_action: JsonValue) -> Candidate:
        with self.worker.hold(f"candidate-{len(self.candidates) + 1:02d}"):
            return super().evaluate(raw_action)

    def finish(
        self, rng: random.Random, *, selected_candidate_id: str | None = None
    ) -> Outcome:
        with self.worker.hold("final"):
            return super().finish(rng, selected_candidate_id=selected_candidate_id)
