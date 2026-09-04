from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from qorl.agent.interface import AgentInterface
from qorl.agent.tools import agent_tools
from qorl.measure.rollout import MAX_CANDIDATES
from qorl.plans.catalog import TaskCatalog
from qorl.plans.fingerprint import plan_sha256
from qorl.plans.schemas import PlanAction
from qorl.plans.verify import compact_plan
from qorl.sft.build_protocol_demo import CALL_SEQUENCE, TASK_ID
from qorl.sft.validate import DemoValidationError, validate_protocol_demo
from qorl.workload.taskset import TaskSet


def raw_plan(tree: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tree, str):
        return {"Node Type": "Seq Scan", "Relation Name": tree, "Alias": tree}
    return {
        "Node Type": "Nested Loop",
        "Plans": [raw_plan(tree["left"]), raw_plan(tree["right"])],
    }


def synthetic_demo(
    repository: Path, candidate_attempts: int = MAX_CANDIDATES
) -> dict[str, Any]:
    task_set = TaskSet.load(repository, "ceb-v1")
    task = next(
        item for item in task_set.inventory["tasks"] if item["task_id"] == TASK_ID
    )
    aliases = sorted(item["alias"] for item in task["relations"])
    indexes = {alias: [] for alias in aliases}
    catalog = TaskCatalog.from_task(task, {alias: set() for alias in aliases})
    leading = {
        "left": {"left": "rt", "right": "ci"},
        "right": {
            "left": "an",
            "right": {"left": "n", "right": {"left": "pi1", "right": "it1"}},
        },
    }
    plan_action = PlanAction.from_raw({"version": 1, "leading": leading}, catalog)
    action, hint = plan_action.to_wire(), plan_action.compile()
    plan = {"Plan": {**raw_plan(leading), "Startup Cost": 1.0}}
    visible_plan = {"Plan": compact_plan(plan["Plan"])}
    tools = agent_tools(aliases)
    maximum_turns = 64
    reserved_decision_turns = candidate_attempts + 1
    inspection_limit = min(len(aliases) * 3, maximum_turns - reserved_decision_turns)
    observation = {
        "task_id": task["task_id"],
        "sql": task_set.load_sql(task),
        "relations": task["relations"],
        "join_edges": task["join_edges"],
        "indexes": indexes,
        "candidate_attempts": candidate_attempts,
        "turn_budget": {
            "total_model_turns": maximum_turns,
            "maximum_inspection_turns": inspection_limit,
            "reserved_final_turns": reserved_decision_turns,
            "reserved_for": {
                "candidate_evaluations": candidate_attempts,
                "finish_or_keep_default": 1,
            },
        },
    }
    interface = AgentInterface(maximum_turns, inspection_limit, observation, tools)
    messages: list[dict[str, Any]] = interface.initial_messages()

    def add(
        turn: int, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> None:
        call_id = f"call-{turn:04d}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, sort_keys=True),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(
                        {**result, "_turn_budget": interface.budget(turn)},
                        sort_keys=True,
                    ),
                },
            ]
        )

    add(1, "get_plan", {"candidate_id": "default"}, visible_plan)
    add(
        2,
        "evaluate_candidate",
        {"action": action},
        {
            "candidate_id": "candidate-01",
            "action_valid": True,
            "constraints_satisfied": True,
            "compiled_hint": hint,
            "plan_sha256": plan_sha256(plan["Plan"]),
        },
    )
    call_sequence = ["get_plan", "evaluate_candidate"]
    if candidate_attempts > 1:
        add(3, "get_plan", {"candidate_id": "candidate-01"}, visible_plan)
        add(4, "finish", {}, {"status": "finished"})
        call_sequence.extend(["get_plan", "finish"])
    else:
        add(3, "finish", {}, {"status": "finished"})
        call_sequence.append("finish")
    return {
        "schema_version": 1,
        "messages": messages,
        "tools": tools,
        "metadata": {
            "task_set_id": "ceb-v1",
            "task_id": task["task_id"],
            "data_identity": task_set.data_identity,
            "runtime_identity": {
                "postgres_image_id": "sha256:test",
                "benchmark_config_id": "benchmark-v2",
            },
            "maximum_model_turns": maximum_turns,
            "call_sequence": call_sequence,
        },
        "evidence": {
            "default_plan": plan,
            "candidates": {
                "candidate-01": {
                    "action": action,
                    "plain_explain": plan,
                    "pg_hint_plan": {
                        "used": "Leading",
                        "not_used": "(none)",
                        "duplicate": "(none)",
                        "error": "(none)",
                    },
                }
            },
        },
    }


class TestProtocolDemo:
    def test_validates_exact_live_protocol_shape(self, repository_root: Path) -> None:
        summary = validate_protocol_demo(
            synthetic_demo(repository_root), repository_root
        )
        assert summary["candidate_ids"] == ["candidate-01"]
        assert summary["terminal_decision"] == "finish"
        assert summary["call_sequence"] == CALL_SEQUENCE

    def test_validates_a_one_candidate_budget(self, repository_root: Path) -> None:
        demo = synthetic_demo(repository_root, candidate_attempts=1)

        summary = validate_protocol_demo(demo, repository_root)

        assert summary["candidate_ids"] == ["candidate-01"]
        assert "up to 1 candidate evaluation" in demo["messages"][0]["content"]

    def test_validates_keep_default_as_a_terminal_decision(
        self, repository_root: Path
    ) -> None:
        demo = synthetic_demo(repository_root)
        observation = json.loads(demo["messages"][1]["content"])
        interface = AgentInterface(
            maximum_model_turns=demo["metadata"]["maximum_model_turns"],
            inspection_turn_limit=observation["turn_budget"][
                "maximum_inspection_turns"
            ],
            observation=observation,
            tools=demo["tools"],
        )
        demo["messages"] = demo["messages"][:4]
        demo["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-0002",
                            "type": "function",
                            "function": {
                                "name": "keep_default",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-0002",
                    "name": "keep_default",
                    "content": json.dumps(
                        {
                            "status": "kept_default",
                            "_turn_budget": interface.budget(2),
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
        demo["metadata"]["call_sequence"] = ["get_plan", "keep_default"]

        summary = validate_protocol_demo(demo, repository_root)

        assert summary["candidate_ids"] == []
        assert summary["terminal_decision"] == "keep_default"

    def test_rejects_unissued_candidate_id(self, repository_root: Path) -> None:
        demo = copy.deepcopy(synthetic_demo(repository_root))
        call = demo["messages"][6]["tool_calls"][0]
        call["function"]["arguments"] = '{"candidate_id":"candidate-02"}'
        with pytest.raises(DemoValidationError, match="was not issued"):
            validate_protocol_demo(demo, repository_root)
