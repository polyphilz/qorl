from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from qorl.action import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    NUMERIC_SETTINGS,
    TaskCatalog,
)
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.tools import agent_tools


TASK = {
    "task_id": "job-test",
    "relations": [
        {"alias": "a", "table": "table_a"},
        {"alias": "b", "table": "table_b"},
    ],
    "join_edges": ["a:table_a.id=b:table_b.a_id"],
}


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "evaluate_candidate",
                                        "arguments": '{"action":{"version":1}}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 5},
            },
        ]

    def models(self) -> dict:
        return {
            "data": [
                {
                    "id": "empero-ai/Qwen3.8-4B-Distill",
                    "max_model_len": 262144,
                }
            ]
        }

    def version(self) -> dict:
        return {"version": "test"}

    def chat(self, body: dict) -> dict:
        self.requests.append(body)
        return self.responses.pop(0)


class FakeEvaluator:
    def __init__(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        repository = Path(self.temporary_directory.name)
        expected = repository / "docker/postgres/benchmark-v1.expected.json"
        expected.parent.mkdir(parents=True)
        expected.write_text(
            json.dumps(
                {
                    "settings": {
                        **{name: "on" for name in BOOLEAN_SETTINGS},
                        **{name: "1" for name in NUMERIC_SETTINGS},
                        **{name: "1" for name in INTEGER_SETTINGS},
                    }
                }
            )
        )
        self.task = TASK
        self.sql = "SELECT 1;"
        self.catalog = TaskCatalog.from_task(
            TASK, {"a": {"table_a_pkey"}, "b": {"table_b_pkey"}}
        )
        self.worker = SimpleNamespace(
            fixture=SimpleNamespace(
                snapshot={"postgresql": {"server_version_num": "180006"}},
                repository=repository,
            )
        )
        self.default = {
            "compact_plan": {"Node Type": "Result"},
            "median_execution_time_ms": 1.0,
        }
        self.timeout_ms = 5_000
        self.candidates = []
        self.actions = []

    def evaluate(self, action: object) -> dict:
        self.actions.append(action)
        candidate = {
            "candidate_id": "candidate-01",
            "action_valid": True,
            "constraints_satisfied": True,
            "compiled_hint": "",
            "duplicate_of": "default",
            "plan_sha256": "abc",
            "compact_plan": {"Node Type": "Result"},
            "provisional_measurements": [
                {"planning_time_ms": 0.1, "execution_time_ms": 1.0}
            ],
            "provisional_speedup": 1.0,
            "errors_or_diagnostics": [],
            "attempts_remaining": 4,
        }
        self.candidates.append(candidate)
        return candidate


def config() -> QoAgentConfig:
    return QoAgentConfig.from_dict(
        {
            "type": "qo_agent",
            "model": "empero-ai/Qwen3.8-4B-Distill",
            "revision": "revision",
            "vllm_version": "test",
            "use_flashinfer_sampler": False,
            "base_url": "http://127.0.0.1:8000/v1",
            "context_length": 262144,
            "maximum_model_turns": 64,
            "request_timeout_seconds": 300,
            "seed": 7,
            "sampling": {"max_tokens": 2048, "temperature": 1.0},
            "thinking": False,
            "tool_call_parser": "qwen3_coder",
        }
    )


class QoAgentTest(unittest.TestCase):
    def test_runs_evaluate_then_finish_tool_loop(self) -> None:
        client = FakeClient()
        policy = QoAgentPolicy(config(), client)
        self.assertEqual(
            policy.preflight()["advertised_models"],
            ["empero-ai/Qwen3.8-4B-Distill"],
        )

        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)
        trace = policy.search(evaluator)  # type: ignore[arg-type]

        self.assertEqual(evaluator.actions, [{"version": 1}])
        self.assertEqual(trace["stop_reason"], "model_finish")
        self.assertEqual(trace["usage"]["prompt_tokens"], 220)
        self.assertEqual(
            trace["initial_observation"]["planner_settings"]["geqo"], "on"
        )
        self.assertEqual(
            trace["initial_observation"]["turn_budget"],
            {
                "total_model_turns": 64,
                "maximum_inspection_turns": 6,
                "reserved_final_turns": 6,
                "reserved_for": {"candidate_evaluations": 5, "finish": 1},
            },
        )
        first_tool_result = json.loads(trace["transcript"][3]["content"])
        self.assertEqual(first_tool_result["_turn_budget"]["turns_remaining"], 63)
        self.assertEqual(len(trace["tools_sha256"]), 64)
        self.assertEqual(
            [message["role"] for message in trace["transcript"]],
            ["system", "user", "assistant", "tool", "assistant", "tool"],
        )
        self.assertEqual(client.requests[0]["tool_choice"], "required")
        first_tools = {
            tool["function"]["name"] for tool in client.requests[0]["tools"]
        }
        second_tools = {
            tool["function"]["name"] for tool in client.requests[1]["tools"]
        }
        self.assertNotIn("finish", first_tools)
        self.assertIn("finish", second_tools)
        self.assertFalse(
            client.requests[0]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_reserved_turns_only_offer_decision_tools(self) -> None:
        client = FakeClient()
        policy = QoAgentPolicy(
            replace(config(), maximum_model_turns=2), client
        )
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        policy.search(evaluator)  # type: ignore[arg-type]

        first_tools = {
            tool["function"]["name"] for tool in client.requests[0]["tools"]
        }
        self.assertEqual(first_tools, {"evaluate_candidate"})

    def test_evaluate_tool_contains_resolvable_join_tree_schema(self) -> None:
        evaluate = next(
            item
            for item in agent_tools(["a", "b"])
            if item["function"]["name"] == "evaluate_candidate"
        )
        parameters = evaluate["function"]["parameters"]
        leading = parameters["properties"]["action"]["properties"]["leading"]
        self.assertEqual(leading["$ref"], "#/$defs/join_node")
        self.assertIn("join_node", parameters["$defs"])
        self.assertIn("join_tree", parameters["$defs"])
        leaves = parameters["$defs"]["join_tree"]["oneOf"][0]
        self.assertEqual(leaves["enum"], ["a", "b"])
        scans = parameters["properties"]["action"]["properties"]["scans"]
        self.assertEqual(
            scans["items"]["properties"]["relation"]["enum"], ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
