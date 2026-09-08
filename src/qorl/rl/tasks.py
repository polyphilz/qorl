from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from qorl.measure.schemas import OutcomeKind
from qorl.paths import REPOSITORY_ROOT
from qorl.rl.schemas import RlRolloutRecord
from qorl.taskset.schemas import TaskSelection
from qorl.taskset.taskset import TaskSet


class QorlTasksetConfig(vf.TasksetConfig):
    repository: Path = REPOSITORY_ROOT
    selection: Path | None = None
    shuffle_seed: int | None = None


class QorlTaskData(vf.TaskData):
    task_id: str
    template_id: str
    rollout_index: int = 0


class QorlTask(vf.Task[QorlTaskData]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def trajectory_reward(self, trace: vf.Trace[vf.TaskData]) -> float:
        record = RlRolloutRecord.model_validate_json(json.dumps(trace.info["qorl"]))
        if record.final is None:
            raise ValueError("unscored rollout failure cannot receive a reward")
        return record.scalar_reward if record.scalar_reward is not None else 0.0

    @vf.metric
    async def rollout_metrics(self, trace: vf.Trace[vf.TaskData]) -> dict[str, float]:
        record = RlRolloutRecord.model_validate_json(json.dumps(trace.info["qorl"]))
        candidates = record.candidates
        final = record.final
        metrics = {
            "candidate_attempts": float(len(candidates)),
            "valid_candidate_attempts": float(
                sum(item.constraints_satisfied for item in candidates)
            ),
            "invalid_candidate_attempts": float(
                sum(
                    not item.constraints_satisfied and not item.execution_timed_out
                    for item in candidates
                )
            ),
            "duplicate_candidate_attempts": float(
                sum(item.duplicate_of is not None for item in candidates)
            ),
            "timeout_candidate_attempts": float(
                sum(item.execution_timed_out for item in candidates)
            ),
            "has_valid_candidate": float(
                any(item.constraints_satisfied for item in candidates)
            ),
            "kept_default": float(
                final is not None and final.kind == OutcomeKind.KEPT_DEFAULT
            ),
            "final_candidate_timeout": float(
                final is not None and final.kind == OutcomeKind.TIMED_OUT
            ),
            "unscored_failure": float(record.failure is not None),
            "model_turns": float(len(trace.calls)),
            "total_tokens": float(trace.num_total_tokens),
            "output_tokens": float(trace.num_output_tokens),
        }
        if final is not None and final.speedup is not None:
            metrics["final_speedup"] = final.speedup
        if record.execution_counts is not None:
            metrics["executions/initial_default"] = float(
                record.execution_counts.initial_default
            )
            metrics["executions/candidate_feedback"] = float(
                record.execution_counts.candidate_feedback
            )
            metrics["executions/final_paired"] = float(
                record.execution_counts.final_paired
            )
        for kind in OutcomeKind:
            metrics[f"outcome/{kind.value}"] = float(
                final is not None and final.kind == kind
            )
        return metrics


class QorlTaskset(vf.Taskset[QorlTask, QorlTasksetConfig]):
    def load(self) -> Iterator[QorlTask]:
        repository = self.config.repository.resolve()
        if self.config.selection is None:
            raise ValueError("RL task loading requires the run's saved selection")
        selection = TaskSelection.model_validate_json(
            (repository / self.config.selection).read_bytes()
        )
        task_set = TaskSet.load(repository, selection.benchmark_id.value)
        tasks = task_set.resolve(selection)
        if self.config.shuffle_seed is not None:
            random.Random(self.config.shuffle_seed).shuffle(tasks)
        for index, task in enumerate(tasks):
            yield QorlTask(
                QorlTaskData(
                    idx=index,
                    name=task.task_id,
                    prompt=task.task_id,
                    task_id=task.task_id,
                    template_id=task.template_id,
                ),
                self.config.task,
            )
