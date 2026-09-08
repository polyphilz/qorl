import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from qorl.measure.schemas import KeptDefaultOutcome, PairedMeasurements, RolloutFailure
from qorl.rl.schemas import RlRolloutRecord
from qorl.rl.tasks import QorlTask, QorlTaskset, QorlTasksetConfig
from qorl.taskset.schemas import BenchmarkId, TaskSelection
from qorl.taskset.taskset import TaskSet


@pytest.mark.parametrize("benchmark", list(BenchmarkId))
def test_loads_exact_saved_selection(
    repository_root: Path, tmp_path: Path, benchmark: BenchmarkId
) -> None:
    catalog = TaskSet.load(repository_root, benchmark.value)
    selected = [catalog.tasks[2], catalog.tasks[0]]
    selection = tmp_path / "tasks.json"
    selection.write_text(
        TaskSelection(
            benchmark_id=benchmark, task_ids=[task.task_id for task in selected]
        ).model_dump_json()
    )
    adapter = QorlTaskset(
        QorlTasksetConfig(id="qorl", selection=selection, repository=repository_root)
    )
    tasks = list(adapter)
    assert [task.key for task in tasks] == [task.task_id for task in selected]
    assert [task.data.template_id for task in tasks] == [
        task.template_id for task in selected
    ]


def test_selection_is_required() -> None:
    with pytest.raises(ValueError, match="saved selection"):
        list(QorlTaskset(QorlTasksetConfig(id="qorl")))


def test_scoring_hooks_use_rl_metadata_and_raw_performance(
    rl_rollout_record: RlRolloutRecord,
) -> None:
    trace = SimpleNamespace(
        calls=[None, None],
        num_total_tokens=100,
        num_output_tokens=10,
        info={"qorl": rl_rollout_record.to_wire()},
    )
    assert inspect.iscoroutinefunction(QorlTask.trajectory_reward)
    assert asyncio.run(QorlTask.trajectory_reward(None, trace)) == 0.15
    metrics = asyncio.run(QorlTask.rollout_metrics(None, trace))
    assert metrics["candidate_attempts"] == 1.0
    assert metrics["has_valid_candidate"] == 1.0
    assert metrics["kept_default"] == 0.0
    assert metrics["final_speedup"] == 2.0
    assert metrics["unscored_failure"] == 0.0


def test_keep_default_does_not_count_as_a_candidate(
    rl_rollout_record: RlRolloutRecord,
) -> None:
    baseline = rl_rollout_record.default
    assert baseline is not None and baseline.timing_reuse_key is not None
    record = RlRolloutRecord.model_validate(
        {
            **rl_rollout_record.model_dump(),
            "candidates": [],
            "final": KeptDefaultOutcome(
                selected_plan_sha256=baseline.plan_sha256,
                timing_reuse_key=baseline.timing_reuse_key,
            ),
            "scalar_reward": None,
        }
    )
    trace = SimpleNamespace(
        calls=[None],
        num_total_tokens=10,
        num_output_tokens=2,
        info={"qorl": record.to_wire()},
    )
    metrics = asyncio.run(QorlTask.rollout_metrics(None, trace))
    assert metrics["has_valid_candidate"] == 0.0
    assert metrics["kept_default"] == 1.0
    assert asyncio.run(QorlTask.trajectory_reward(None, trace)) == 0.0


def test_failure_cannot_receive_scalar_reward(
    rl_rollout_record: RlRolloutRecord,
) -> None:
    record = RlRolloutRecord.model_validate(
        {
            **rl_rollout_record.model_dump(),
            "final": None,
            "failure": RolloutFailure(
                operation="default",
                error_type="QueryTimeoutError",
                error="timeout",
                paired=PairedMeasurements(),
            ),
            "scalar_reward": None,
        }
    )
    trace = SimpleNamespace(
        calls=[],
        num_total_tokens=0,
        num_output_tokens=0,
        info={"qorl": record.to_wire()},
    )
    with pytest.raises(ValueError, match="unscored"):
        asyncio.run(QorlTask.trajectory_reward(None, trace))
    metrics = asyncio.run(QorlTask.rollout_metrics(None, trace))
    assert metrics["unscored_failure"] == 1.0
    assert "final_speedup" not in metrics
