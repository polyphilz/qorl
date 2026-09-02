from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from qorl.workload.taskset import TaskSet


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
    repository: Path = Path(".")
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
    async def trajectory_reward(self, trace: vf.Trace) -> float:
        return float(trace.info["qorl"]["final"]["trajectory_reward"])

    @vf.metric
    async def rollout_metrics(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info["qorl"]
        candidates = result["candidates"]
        final = result["final"]
        return {
            "candidate_attempts": float(len(candidates)),
            "valid_candidate_attempts": float(
                sum(candidate["constraints_satisfied"] for candidate in candidates)
            ),
            "invalid_candidate_attempts": float(
                sum(not candidate["constraints_satisfied"] for candidate in candidates)
            ),
            "duplicate_candidate_attempts": float(
                sum(candidate["duplicate_of"] is not None for candidate in candidates)
            ),
            "timeout_candidate_attempts": float(
                sum(
                    candidate.get("execution_timed_out", False)
                    for candidate in candidates
                )
            ),
            "has_valid_candidate": float(
                final["status"] == "completed"
                and final.get("decision") != "keep_default"
            ),
            "kept_default": float(final.get("decision") == "keep_default"),
            "final_candidate_timeout": float(
                final["status"] == "candidate_timeout"
            ),
            "final_speedup": float(final.get("score", 0.0)),
            "model_turns": float(len(trace.calls)),
            "total_tokens": float(trace.num_total_tokens),
            "output_tokens": float(trace.num_output_tokens),
        }


class QorlTaskset(vf.Taskset[QorlTask, QorlTasksetConfig]):
    def load(self) -> Iterator[QorlTask]:
        repository = self.config.repository.resolve()
        task_set = TaskSet.load(repository, "ceb-v1")
        selection_path = (
            self.config.selection
            if self.config.selection.is_absolute()
            else repository / self.config.selection
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selected_items(selection, self.config.split)
        tasks = {task["task_id"]: task for task in task_set.inventory["tasks"]}
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
