from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from qorl.measure.schemas import OutcomeKind
from qorl.paths import REPOSITORY_ROOT
from qorl.rl.schemas import RlRolloutRecord
from qorl.taskset.taskset import TaskSet

SELECTION_SPLITS = {
    "qorl-rl-pilot-v1": {"spike", "train", "validation"},
    "qorl-rl-run-v2": {"train"},
}


def selected_items(selection: dict, split: str) -> list[dict[str, str]]:
    inventory_id = selection.get("inventory_id")
    if inventory_id not in SELECTION_SPLITS:
        raise ValueError(f"unexpected QORL RL inventory: {inventory_id}")
    if split not in SELECTION_SPLITS[inventory_id]:
        raise ValueError(f"split {split!r} is not allowed by {inventory_id}")
    return selection["splits"][split]


class QorlTasksetConfig(vf.TasksetConfig):
    repository: Path = REPOSITORY_ROOT
    selection: Path = Path("experiments/003-rl-pilot-v1/selection.json")
    split: Literal["spike", "train", "validation"] = "spike"


class QorlTaskData(vf.TaskData):
    task_id: str
    template_id: str


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
        return metrics


class QorlTaskset(vf.Taskset[QorlTask, QorlTasksetConfig]):
    def load(self) -> Iterator[QorlTask]:
        repository = self.config.repository.resolve()
        task_set = TaskSet.load(repository, "ceb")
        selection_path = (
            self.config.selection
            if self.config.selection.is_absolute()
            else repository / self.config.selection
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selected_items(selection, self.config.split)
        tasks = {task.task_id: task.model_dump() for task in task_set.tasks}
        for index, item in enumerate(selected):
            task = tasks[item["task_id"]]
            expected_partition = (
                "validation" if self.config.split == "validation" else "train"
            )
            if task["partition"] != expected_partition:
                raise ValueError(f"task in wrong partition: {task['task_id']}")
            yield QorlTask(
                QorlTaskData(
                    idx=index,
                    name=task["task_id"],
                    prompt=task["task_id"],
                    task_id=task["task_id"],
                    template_id=task["template_id"],
                ),
                self.config.task,
            )
