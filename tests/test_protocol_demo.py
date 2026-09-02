from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from qorl.action import TaskCatalog, compile_action
from qorl.agent.prompts import SYSTEM_PROMPT
from qorl.agent.protocol import AgentProtocol, RESERVED_DECISION_TURNS
from qorl.agent.tools import agent_tools
from qorl.calibration import plan_sha256
from qorl.fixture import TaskSet
from qorl.rollout import MAX_CANDIDATES
from scripts.sft.build_protocol_demo import CALL_SEQUENCE, TASK_ID
from scripts.utils.protocol_demo import DemoValidationError, validate_protocol_demo


ROOT = Path(__file__).resolve().parents[1]


def raw_plan(tree: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tree, str):
        return {"Node Type": "Seq Scan", "Relation Name": tree, "Alias": tree}
    return {
        "Node Type": "Nested Loop",
        "Plans": [raw_plan(tree["left"]), raw_plan(tree["right"])],
    }


def synthetic_demo() -> dict[str, Any]:
    task_set = TaskSet.load(ROOT, "ceb-v1")
    task = next(item for item in task_set.inventory["tasks"] if item["task_id"] == TASK_ID)
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
    action, hint = compile_action({"version": 1, "leading": leading}, catalog)
    plan = {"Plan": raw_plan(leading)}
    tools = agent_tools(aliases)
    maximum_turns = 64
    inspection_limit = min(
        len(aliases) * 3, maximum_turns - RESERVED_DECISION_TURNS
    )
    observation = {
        "task_id": task["task_id"],
        "sql": task_set.load_sql(task),
        "relations": task["relations"],
        "join_edges": task["join_edges"],
        "indexes": indexes,
        "turn_budget": {
            "total_model_turns": maximum_turns,
            "maximum_inspection_turns": inspection_limit,
            "reserved_final_turns": RESERVED_DECISION_TURNS,
            "reserved_for": {
                "candidate_evaluations": MAX_CANDIDATES,
                "finish_or_keep_default": 1,
            },
        },
    }
    protocol = AgentProtocol(maximum_turns, inspection_limit, observation, tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(observation, sort_keys=True)},
    ]

    def add(turn: int, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
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
                        {**result, "_turn_budget": protocol.budget(turn)},
                        sort_keys=True,
                    ),
                },
            ]
        )

    add(1, "get_plan", {"candidate_id": "default"}, plan)
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
    add(3, "get_plan", {"candidate_id": "candidate-01"}, plan)
    add(4, "finish", {}, {"status": "finished"})
    return {
        "schema_version": 1,
        "messages": messages,
        "tools": tools,
        "metadata": {
            "task_set_id": "ceb-v1",
            "task_id": task["task_id"],
            "database": task_set.inventory["database"],
            "maximum_model_turns": maximum_turns,
            "call_sequence": CALL_SEQUENCE,
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


class ProtocolDemoTest(unittest.TestCase):
    def test_validates_exact_live_protocol_shape(self) -> None:
        summary = validate_protocol_demo(synthetic_demo(), ROOT)
        self.assertEqual(summary["candidate_ids"], ["candidate-01"])
        self.assertEqual(summary["terminal_decision"], "finish")
        self.assertEqual(summary["call_sequence"], CALL_SEQUENCE)

    def test_validates_keep_default_as_a_terminal_decision(self) -> None:
        demo = synthetic_demo()
        observation = json.loads(demo["messages"][1]["content"])
        protocol = AgentProtocol(
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
                            "_turn_budget": protocol.budget(2),
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
        demo["metadata"]["call_sequence"] = ["get_plan", "keep_default"]

        summary = validate_protocol_demo(demo, ROOT)

        self.assertEqual(summary["candidate_ids"], [])
        self.assertEqual(summary["terminal_decision"], "keep_default")

    def test_rejects_unissued_candidate_id(self) -> None:
        demo = copy.deepcopy(synthetic_demo())
        call = demo["messages"][6]["tool_calls"][0]
        call["function"]["arguments"] = '{"candidate_id":"candidate-02"}'
        with self.assertRaisesRegex(DemoValidationError, "was not issued"):
            validate_protocol_demo(demo, ROOT)


if __name__ == "__main__":
    unittest.main()
