"""Measurement affinity and cancellation with real pool claims and fake SQL."""

import json
import random
import subprocess
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock, Timer

import pytest
from tests.qorl.measure.test_rollout import ACTION, PLAN, TASK, Sql

from qorl.measure.schemas import RolloutMeasurementSettings
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import WorkerAllocation
from qorl.rl.runtime import QorlRuntime
from qorl.rl.workers import PooledExecutor, PooledRolloutEvaluator
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig


@pytest.fixture
def pool(
    repository_root: Path, pool_config: PoolConfig, postgres_config: PostgresConfig
) -> QorlRuntime:
    pool = QorlRuntime(
        TaskSet.load(repository_root, "job"), pool_config, "leased", postgres_config
    )
    pool.indexes.by_table.update(table_a=frozenset(), table_b=frozenset())
    return pool


def assert_pool_released(pool: QorlRuntime) -> None:
    cancelled = Event()
    timeout = Timer(2, cancelled.set)
    timeout.start()
    try:
        with ExitStack() as owners:
            slots = [
                owners.enter_context(pool.claim_worker(cancel=cancelled))
                for _ in pool.workers
            ]
            assert len({slot.resources.index for slot in slots}) == len(pool.workers)
    finally:
        timeout.cancel()


def test_phases_keep_warmups_and_final_pairs_on_one_worker(pool: QorlRuntime) -> None:
    calls: list[tuple[int, bool, bool]] = []

    def execute(
        slot: int, command: list[str], query: str
    ) -> subprocess.CompletedProcess[str]:
        if not query.startswith("EXPLAIN"):
            calls.append((slot, False, False))
            return subprocess.CompletedProcess(command, 0, "catalog", "")
        analyze = "EXPLAIN (ANALYZE" in query
        candidate = "Set(seq_page_cost" in query
        calls.append((slot, analyze, candidate))
        document = {
            "Plan": PLAN,
            "Execution Time": (slot + 1) * (5 if candidate else 10),
            "Planning Time": 1,
        }
        diagnostics = (
            "HintStateDump: {used hints:Set(seq_page_cost)}, {not used hints:(none)}, "
            "{duplicate hints:(none)}, {error hints:(none)}"
            if candidate
            else ""
        )
        return subprocess.CompletedProcess(
            command, 0, json.dumps([document]), diagnostics
        )

    for slot in pool.workers:
        index = slot.resources.index

        def sql(
            command: list[str], query: str, index: int = index
        ) -> subprocess.CompletedProcess[str]:
            return execute(index, command, query)

        slot.client = PostgresClient(sql, pool.settings, pool.indexes)
        slot.client.allocation = WorkerAllocation(
            cpuset=slot.resources.cpuset,
            physical_core_count=slot.resources.physical_core_count,
            memory_bytes=slot.resources.memory_bytes,
        )
    executor = PooledExecutor(pool, Event())
    evaluator = PooledRolloutEvaluator(
        executor,
        Sql(),
        TASK,
        measurement=RolloutMeasurementSettings(
            default_warmups=1,
            default_measurements=3,
            candidate_feedback_warmups=1,
            candidate_feedback_measurements=1,
            paired_warmups=1,
            paired_measurements=3,
            default_timeout_seconds=300,
            candidate_timeout_floor_seconds=5,
            candidate_timeout_multiplier=3,
        ),
        max_candidates=2,
    )
    baseline = evaluator.start()
    assert baseline.median_execution_time_ms == 10
    assert executor.allocation == pool.workers[0].client.allocation
    assert executor.admin_sql("SELECT 'catalog'") == "catalog"
    candidate = evaluator.evaluate(ACTION)
    assert candidate.selection_eligible and candidate.execution_feedback is not None
    assert candidate.execution_feedback.measurements[0].execution_time_ms == 15
    final = evaluator.finish(
        random.Random(0), selected_candidate_id=candidate.candidate_id
    )
    assert final.kind == "measured" and final.speedup == 2
    assert final.default_median_execution_time_ms == 40
    assert final.candidate_median_execution_time_ms == 20
    assert [slot for slot, _, _ in calls] == [0] * 5 + [1] + [2] * 3 + [3] * 8
    assert [(lease.operation, lease.worker_slot) for lease in executor.leases] == [
        ("initial_default", 0),
        ("inspection", 1),
        ("candidate-01", 2),
        ("final", 3),
    ]
    assert_pool_released(pool)


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_failed_phase_releases_its_worker(pool: QorlRuntime, failure: str) -> None:
    cancelled = Event()
    executor = PooledExecutor(pool, cancelled)
    expected = PostgresError if failure == "error" else CancelledError
    with pytest.raises(expected), executor.hold("initial_default"):
        if failure == "error":
            raise PostgresError("SQL failed")
        cancelled.set()
        executor.admin_sql("must not run")
    assert len(executor.leases) == 1
    assert_pool_released(pool)


def test_cancellation_while_waiting_does_not_need_an_available_worker(
    pool: QorlRuntime,
) -> None:
    cancelled, waiting = Event(), Event()
    executor = PooledExecutor(pool, cancelled)

    def query() -> None:
        waiting.set()
        executor.admin_sql("must not run")

    with ExitStack() as owners:
        for _ in pool.workers:
            owners.enter_context(pool.claim_worker())
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(query)
            assert waiting.wait(2)
            cancelled.set()
            with pytest.raises(CancelledError):
                pending.result(timeout=2)
        assert not executor.leases
    assert_pool_released(pool)


def test_twelve_episodes_never_overlap_on_a_database_worker(pool: QorlRuntime) -> None:
    active: set[int] = set()
    peak = 0
    lock = Lock()
    together = Barrier(len(pool.workers), timeout=3)

    def measure(_: int) -> None:
        nonlocal peak
        executor = PooledExecutor(pool, Event())
        with executor.hold("initial_default") as client:
            slot = next(
                slot.resources.index for slot in pool.workers if slot.client is client
            )
            with lock:
                assert slot not in active
                active.add(slot)
                peak = max(peak, len(active))
            together.wait()
            with lock:
                active.remove(slot)
        assert len(executor.leases) == 1

    with ThreadPoolExecutor(max_workers=12) as threads:
        list(threads.map(measure, range(12)))
    assert peak == len(pool.workers) and not active
    assert_pool_released(pool)


def test_measurement_leases_reject_different_worker_capacities(
    pool: QorlRuntime,
) -> None:
    slot = pool.workers[1]
    slot.resources = replace(
        slot.resources, memory_bytes=slot.resources.memory_bytes * 2
    )
    with pytest.raises(ValueError, match="uniform worker resource limits"):
        PooledExecutor(pool, Event())
