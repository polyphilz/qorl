from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from qorl.plans.action import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    NUMERIC_SETTINGS,
    TaskCatalog,
)
from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import OpenAIModelClient
from qorl.agent.protocol import AgentProtocol
from qorl.agent.tool_runtime import AgentEnvironment
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


class KeepDefaultClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
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
                                        "name": "keep_default",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            }
        ]


class FakeLoraClient(FakeClient):
    def models(self) -> dict:
        return {
            "data": [
                {"id": "qorl-base", "max_model_len": 262144},
                {
                    "id": "qorl-protocol-adapter",
                    "parent": "qorl-base",
                    "max_model_len": None,
                },
            ]
        }


class ContextLimitClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.responses[0]["usage"] = {
            "prompt_tokens": 18_500,
            "completion_tokens": 500,
            "total_tokens": 19_000,
        }


class MissingUsageClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.responses[0].pop("usage")


class FakeEvaluator:
    def __init__(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        repository = Path(self.temporary_directory.name)
        self.requested_settings: set[str] = set()

        def settings(names: set[str]) -> dict[str, str]:
            self.requested_settings = names
            return {
                **{name: "on" for name in BOOLEAN_SETTINGS},
                **{name: "1" for name in NUMERIC_SETTINGS},
                **{name: "1" for name in INTEGER_SETTINGS},
            }
        self.task = TASK
        self.sql = "SELECT 1;"
        self.catalog = TaskCatalog.from_task(
            TASK, {"a": {"table_a_pkey"}, "b": {"table_b_pkey"}}
        )
        self.worker = SimpleNamespace(
            settings=settings,
            fixture=SimpleNamespace(
                snapshot={"postgresql": {"server_version_num": "180006"}},
                repository=repository,
            )
        )
        self.default = {
            "compact_plan": {"Node Type": "Result"},
            "plain_explain": {
                "Plan": {
                    "Node Type": "Result",
                    "Plan Rows": 1,
                    "Total Cost": 99.0,
                }
            },
            "median_execution_time_ms": 1.0,
        }
        self.timeout_ms = 5_000
        self.candidates = []
        self.actions = []
        self.kept_default = False

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

    def keep_default(self) -> dict[str, str]:
        if self.candidates:
            raise RuntimeError(
                "keep_default must be selected before submitting a candidate"
            )
        self.kept_default = True
        return {"status": "kept_default"}


def config() -> QoAgentConfig:
    return QoAgentConfig.from_dict(
        {
            "type": "qo_agent",
            "model": "empero-ai/Qwen3.8-4B-Distill",
            "revision": "revision",
            "vllm_version": "test",
            "use_flashinfer_sampler": False,
            "enable_prefix_caching": True,
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
    def test_model_visible_request_bytes_are_stable(self) -> None:
        client = FakeClient()
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        QoAgentPolicy(config(), client).search(evaluator)  # type: ignore[arg-type]

        digests = [
            hashlib.sha256(json.dumps(request).encode("utf-8")).hexdigest()
            for request in client.requests
        ]
        self.assertEqual(
            digests,
            [
                "7bb5e0267eb6ad3f602c2e5c6d163f72ef4cc612e1d0ea99c5a97831a3d6b961",
                "e3a41ca9fb08dd61d09a101a29e4e5846b6fc673aedb3d8cc3a18a4730370b4e",
            ],
        )

    def test_protocol_exposes_a_one_candidate_training_budget(self) -> None:
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)
        evaluator.max_candidates = 1

        protocol = AgentProtocol.from_evaluator(evaluator, 64)

        self.assertEqual(protocol.observation["candidate_attempts"], 1)
        self.assertEqual(protocol.observation["turn_budget"]["reserved_final_turns"], 2)
        self.assertEqual(protocol.available_tool_names(1, 1), {"finish"})

    def test_get_default_plan_returns_only_compact_fields(self) -> None:
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        result, finished = AgentEnvironment(evaluator).execute(
            "get_plan", {"candidate_id": "default"}
        )

        self.assertEqual(result, {"Plan": {"Node Type": "Result", "Plan Rows": 1}})
        self.assertFalse(finished)

    def test_exhausted_candidate_budget_is_tool_feedback(self) -> None:
        evaluator = FakeEvaluator()

        def exhausted(_: object) -> dict:
            raise RuntimeError("rollout candidate budget is exhausted")

        evaluator.evaluate = exhausted
        result, finished = AgentEnvironment(evaluator).execute(
            "evaluate_candidate", {"action": {"version": 1}}
        )

        self.assertEqual(result, {"error": "rollout candidate budget is exhausted"})
        self.assertFalse(finished)
        evaluator.temporary_directory.cleanup()

    def test_model_client_accepts_a_rollout_scoped_api_key(self) -> None:
        client = OpenAIModelClient("http://example.test/v1", 10, "secret")

        self.assertEqual(client.api_key, "secret")

    def test_request_seed_can_be_left_to_the_rollout_server(self) -> None:
        policy = QoAgentPolicy(replace(config(), seed=None), FakeClient())
        body = policy.request_body([], [], "job-test", 1)

        self.assertNotIn("seed", body)

    def test_preflight_inherits_lora_context_length_from_parent(self) -> None:
        policy = QoAgentPolicy(
            replace(config(), model="qorl-protocol-adapter"), FakeLoraClient()
        )

        identity = policy.preflight()

        self.assertEqual(identity["model"]["id"], "qorl-protocol-adapter")
        self.assertEqual(identity["effective_context_model"]["id"], "qorl-base")

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
            trace["initial_observation"]["planner_settings"]["enable_hashjoin"],
            "on",
        )
        self.assertEqual(
            evaluator.requested_settings,
            set(BOOLEAN_SETTINGS) | set(NUMERIC_SETTINGS) | set(INTEGER_SETTINGS),
        )
        self.assertEqual(
            trace["initial_observation"]["turn_budget"],
            {
                "total_model_turns": 64,
                "maximum_inspection_turns": 6,
                "reserved_final_turns": 6,
                "reserved_for": {
                    "candidate_evaluations": 5,
                    "finish_or_keep_default": 1,
                },
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
        first_tools = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
        second_tools = {
            tool["function"]["name"] for tool in client.requests[1]["tools"]
        }
        self.assertNotIn("finish", first_tools)
        self.assertIn("keep_default", first_tools)
        self.assertIn("finish", second_tools)
        self.assertNotIn("keep_default", second_tools)
        self.assertFalse(client.requests[0]["chat_template_kwargs"]["enable_thinking"])

    def test_keep_default_ends_without_a_candidate(self) -> None:
        client = KeepDefaultClient()
        policy = QoAgentPolicy(config(), client)
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        trace = policy.search(evaluator)  # type: ignore[arg-type]

        self.assertEqual(trace["stop_reason"], "model_keep_default")
        self.assertTrue(evaluator.kept_default)
        self.assertEqual(evaluator.candidates, [])
        self.assertEqual(trace["tool_events"][0]["result"]["status"], "kept_default")

    def test_terminal_tools_enforce_the_decision_order(self) -> None:
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)
        environment = AgentEnvironment(evaluator)

        result, finished = environment.execute("finish", {})
        self.assertIn("use keep_default", result["error"])
        self.assertFalse(finished)

        evaluator.evaluate({"version": 1})
        result, finished = environment.execute("keep_default", {})
        self.assertIn("before submitting a candidate", result["error"])
        self.assertFalse(finished)

    def test_reserved_turns_only_offer_decision_tools(self) -> None:
        client = FakeClient()
        policy = QoAgentPolicy(replace(config(), maximum_model_turns=2), client)
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        policy.search(evaluator)  # type: ignore[arg-type]

        first_tools = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
        self.assertEqual(first_tools, {"evaluate_candidate", "keep_default"})

    def test_stops_before_the_next_turn_would_exceed_context(self) -> None:
        client = ContextLimitClient()
        policy = QoAgentPolicy(replace(config(), context_length=20_480), client)
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        trace = policy.search(evaluator)  # type: ignore[arg-type]

        self.assertEqual(trace["stop_reason"], "context_budget")
        self.assertEqual(len(client.requests), 1)
        self.assertGreater(trace["context_estimate_tokens"], 19_000)

    def test_stops_if_the_server_omits_token_usage(self) -> None:
        client = MissingUsageClient()
        policy = QoAgentPolicy(config(), client)
        evaluator = FakeEvaluator()
        self.addCleanup(evaluator.temporary_directory.cleanup)

        trace = policy.search(evaluator)  # type: ignore[arg-type]

        self.assertEqual(trace["stop_reason"], "missing_token_usage")
        self.assertEqual(len(client.requests), 1)

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
        self.assertEqual(scans["items"]["properties"]["relation"]["enum"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
