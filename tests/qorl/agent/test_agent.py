from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError, ModelRequestError, OpenAIModelClient
from qorl.agent.interface import AgentInterface
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import Baseline, Candidate, Measurement, MeasurementStatus
from qorl.plans.catalog import TaskCatalog
from qorl.plans.schemas import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    NUMERIC_SETTINGS,
)

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


class FakeEvaluator(RolloutEvaluator[InspectionExecutor]):
    def __init__(self, repository: Path) -> None:
        self.requested_settings: set[str] = set()

        def settings(names: set[str]) -> dict[str, str]:
            self.requested_settings = names
            return {
                **dict.fromkeys(BOOLEAN_SETTINGS, "on"),
                **dict.fromkeys(NUMERIC_SETTINGS, "1"),
                **dict.fromkeys(INTEGER_SETTINGS, "1"),
            }

        def admin_sql(sql: str) -> str:
            del sql
            return ""

        self.task = TASK
        self.sql = "SELECT 1;"
        self.catalog = TaskCatalog.from_task(
            TASK, {"a": {"table_a_pkey"}, "b": {"table_b_pkey"}}
        )
        self._worker = SimpleNamespace(
            settings=settings,
            admin_sql=admin_sql,
            fixture=SimpleNamespace(
                snapshot={"postgresql": {"server_version_num": "180006"}},
                repository=repository,
            ),
        )
        self.default = Baseline(
            plan_sha256="default",
            compact_plan={"Node Type": "Result"},
            plain_explain={
                "Plan": {
                    "Node Type": "Result",
                    "Plan Rows": 1,
                    "Total Cost": 99.0,
                }
            },
            median_execution_time_ms=1.0,
        )
        self.timeout_ms = 5_000
        self.candidates = []
        self.actions = []
        self.kept_default = False

    def evaluate(self, action: object) -> Candidate:
        self.actions.append(action)
        candidate = Candidate(
            candidate_id="candidate-01",
            action=action,
            action_valid=True,
            constraints_satisfied=True,
            compiled_hint="",
            duplicate_of="default",
            plan_sha256="abc",
            compact_plan={"Node Type": "Result"},
            provisional_measurements=[
                Measurement(
                    planning_time_ms=0.1,
                    execution_time_ms=1.0,
                    plan_sha256="abc",
                )
            ],
            provisional_speedup=1.0,
            errors_or_diagnostics=[],
            pg_hint_plan=None,
            attempts_remaining=4,
        )
        self.candidates.append(candidate)
        return candidate

    def keep_default(self) -> dict[str, str]:
        if self.candidates:
            raise RuntimeError(
                "keep_default must be selected before submitting a candidate"
            )
        self.kept_default = True
        return {"status": "kept_default"}


class ValidationOnlyEvaluator(FakeEvaluator):
    def evaluate(self, action: object) -> Candidate:
        candidate = (
            super()
            .evaluate(action)
            .model_copy(
                update={
                    "provisional_measurements": [],
                    "provisional_speedup": None,
                    "measurement_status": MeasurementStatus.NOT_MEASURED,
                }
            )
        )
        self.candidates[-1] = candidate
        return candidate


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


class TestQoAgent:
    def test_model_visible_request_bytes_are_stable(self, tmp_path: Path) -> None:
        client = FakeClient()
        evaluator = FakeEvaluator(tmp_path)

        QoAgentPolicy(config(), client).search(evaluator)

        digests = [
            hashlib.sha256(json.dumps(request).encode("utf-8")).hexdigest()
            for request in client.requests
        ]
        assert digests == [
            "ace720366597c57187b65fb79cb141b61b98c2e71743f6dd9cfb755ecc8e6221",
            "d81683bfe25034d7279d70efcab6b66c32e6ef696d93776516b2dc3bf477e3c6",
        ]

    def test_protocol_exposes_a_one_candidate_training_budget(
        self, tmp_path: Path
    ) -> None:
        evaluator = FakeEvaluator(tmp_path)
        evaluator.max_candidates = 1

        interface = AgentInterface.from_evaluator(evaluator, 64)

        assert interface.observation["candidate_attempts"] == 1
        assert interface.observation["turn_budget"]["reserved_final_turns"] == 2
        assert interface.available_tool_names(1, 1) == {"finish"}
        prompt = interface.initial_messages()[0]["content"]
        assert "up to 1 candidate evaluation plus one terminal decision" in prompt

    def test_get_default_plan_returns_only_compact_fields(self, tmp_path: Path) -> None:
        evaluator = FakeEvaluator(tmp_path)

        result, finished = AgentEnvironment(evaluator).execute(
            "get_plan", {"candidate_id": "default"}
        )

        assert result == {"Plan": {"Node Type": "Result", "Plan Rows": 1}}
        assert not finished

    def test_exhausted_candidate_budget_is_tool_feedback(self, tmp_path: Path) -> None:
        evaluator = FakeEvaluator(tmp_path)

        def exhausted(_: object) -> dict:
            raise RuntimeError("rollout candidate budget is exhausted")

        evaluator.evaluate = exhausted
        result, finished = AgentEnvironment(evaluator).execute(
            "evaluate_candidate", {"action": {"version": 1}}
        )

        assert result == {"error": "rollout candidate budget is exhausted"}
        assert not finished

    def test_model_client_accepts_a_rollout_scoped_api_key(self) -> None:
        client = OpenAIModelClient("http://example.test/v1", 10, "secret")

        assert client.api_key == "secret"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_model_client_treats_non_rate_limit_4xx_as_fatal(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        error = urllib.error.HTTPError(
            "http://example.test/v1/chat/completions",
            status,
            "rejected",
            None,
            io.BytesIO(b'{"error":"bad request"}'),
        )

        def reject(_request: urllib.request.Request, timeout: int) -> None:
            del timeout
            raise error

        monkeypatch.setattr(urllib.request, "urlopen", reject)

        with pytest.raises(ModelRequestError, match=f"HTTP {status}"):
            OpenAIModelClient("http://example.test/v1", 10).chat({})

    @pytest.mark.parametrize("status", [429, 500])
    def test_model_client_leaves_retryable_http_failures_as_model_errors(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        error = urllib.error.HTTPError(
            "http://example.test/v1/chat/completions",
            status,
            "temporary",
            None,
            io.BytesIO(b'{"error":"retry"}'),
        )

        def reject(_request: urllib.request.Request, timeout: int) -> None:
            del timeout
            raise error

        monkeypatch.setattr(urllib.request, "urlopen", reject)

        with pytest.raises(ModelError) as caught:
            OpenAIModelClient("http://example.test/v1", 10).chat({})
        assert not isinstance(caught.value, ModelRequestError)

    def test_request_seed_can_be_left_to_the_rollout_server(self) -> None:
        policy = QoAgentPolicy(replace(config(), seed=None), FakeClient())
        body = policy.request_body([], [], "job-test", 1)

        assert "seed" not in body

    def test_preflight_inherits_lora_context_length_from_parent(self) -> None:
        policy = QoAgentPolicy(
            replace(config(), model="qorl-protocol-adapter"), FakeLoraClient()
        )

        identity = policy.preflight()

        assert identity["model"]["id"] == "qorl-protocol-adapter"
        assert identity["effective_context_model"]["id"] == "qorl-base"

    def test_runs_evaluate_then_finish_tool_loop(self, tmp_path: Path) -> None:
        client = FakeClient()
        policy = QoAgentPolicy(config(), client)
        assert policy.preflight()["advertised_models"] == [
            "empero-ai/Qwen3.8-4B-Distill"
        ]

        evaluator = FakeEvaluator(tmp_path)
        trace = policy.search(evaluator)

        assert evaluator.actions == [{"version": 1}]
        assert trace["stop_reason"] == "model_finish"
        assert trace["usage"]["prompt_tokens"] == 220
        assert (
            trace["initial_observation"]["planner_settings"]["enable_hashjoin"] == "on"
        )
        assert evaluator.requested_settings == set(BOOLEAN_SETTINGS) | set(
            NUMERIC_SETTINGS
        ) | set(INTEGER_SETTINGS)
        assert trace["initial_observation"]["turn_budget"] == {
            "total_model_turns": 64,
            "maximum_inspection_turns": 6,
            "reserved_final_turns": 6,
            "reserved_for": {
                "candidate_evaluations": 5,
                "finish_or_keep_default": 1,
            },
        }
        first_tool_result = json.loads(trace["transcript"][3]["content"])
        assert first_tool_result["_turn_budget"]["turns_remaining"] == 63
        assert len(trace["tools_sha256"]) == 64
        assert [message["role"] for message in trace["transcript"]] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert client.requests[0]["tool_choice"] == "required"
        first_tools = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
        second_tools = {
            tool["function"]["name"] for tool in client.requests[1]["tools"]
        }
        assert "finish" not in first_tools
        assert "keep_default" in first_tools
        assert "finish" in second_tools
        assert "keep_default" not in second_tools
        assert not client.requests[0]["chat_template_kwargs"]["enable_thinking"]

    def test_validation_only_candidate_prints_without_a_speedup(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        evaluator = ValidationOnlyEvaluator(tmp_path)

        trace = QoAgentPolicy(config(), FakeClient()).search(evaluator)

        assert trace["stop_reason"] == "model_finish"
        assert "candidate-01: validated" in capsys.readouterr().out

    def test_keep_default_ends_without_a_candidate(self, tmp_path: Path) -> None:
        client = KeepDefaultClient()
        policy = QoAgentPolicy(config(), client)
        evaluator = FakeEvaluator(tmp_path)

        trace = policy.search(evaluator)

        assert trace["stop_reason"] == "model_keep_default"
        assert evaluator.kept_default
        assert evaluator.candidates == []
        assert trace["tool_events"][0]["result"]["status"] == "kept_default"

    def test_terminal_tools_enforce_the_decision_order(self, tmp_path: Path) -> None:
        evaluator = FakeEvaluator(tmp_path)
        environment = AgentEnvironment(evaluator)

        result, finished = environment.execute("finish", {})
        assert "use keep_default" in result["error"]
        assert not finished

        evaluator.evaluate({"version": 1})
        result, finished = environment.execute("keep_default", {})
        assert "before submitting a candidate" in result["error"]
        assert not finished

    def test_reserved_turns_only_offer_decision_tools(self, tmp_path: Path) -> None:
        client = FakeClient()
        policy = QoAgentPolicy(replace(config(), maximum_model_turns=2), client)
        evaluator = FakeEvaluator(tmp_path)

        policy.search(evaluator)

        first_tools = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
        assert first_tools == {"evaluate_candidate", "keep_default"}

    def test_stops_before_the_next_turn_would_exceed_context(
        self, tmp_path: Path
    ) -> None:
        client = ContextLimitClient()
        policy = QoAgentPolicy(replace(config(), context_length=20_480), client)
        evaluator = FakeEvaluator(tmp_path)

        trace = policy.search(evaluator)

        assert trace["stop_reason"] == "context_budget"
        assert len(client.requests) == 1
        assert trace["context_estimate_tokens"] > 19_000

    def test_stops_if_the_server_omits_token_usage(self, tmp_path: Path) -> None:
        client = MissingUsageClient()
        policy = QoAgentPolicy(config(), client)
        evaluator = FakeEvaluator(tmp_path)

        trace = policy.search(evaluator)

        assert trace["stop_reason"] == "missing_token_usage"
        assert len(client.requests) == 1

    def test_evaluate_tool_contains_resolvable_join_tree_schema(self) -> None:
        evaluate = next(
            item
            for item in agent_tools(["a", "b"])
            if item["function"]["name"] == "evaluate_candidate"
        )
        parameters = evaluate["function"]["parameters"]
        leading = parameters["properties"]["action"]["properties"]["leading"]
        assert leading["$ref"] == "#/$defs/JoinNode"
        join_node = parameters["$defs"]["JoinNode"]
        assert join_node["properties"]["left"]["anyOf"][1]["$ref"] == "#/$defs/JoinNode"
        assert join_node["properties"]["left"]["anyOf"][0]["enum"] == ["a", "b"]
        scans = parameters["$defs"]["ScanConstraint"]
        assert scans["properties"]["relation"]["enum"] == ["a", "b"]
