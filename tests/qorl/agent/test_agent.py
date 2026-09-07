from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import ModelError, ModelRequestError, OpenAIModelClient
from qorl.agent.interface import AgentInterface
from qorl.agent.schemas import AgentSettings
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import Baseline, Candidate, Measurement, MeasurementStatus
from qorl.measure.validation import PlanValidationEvaluator
from qorl.plans.catalog import TaskCatalog
from qorl.plans.schemas import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    NUMERIC_SETTINGS,
)
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import ExplainResult, PostgresIndexes, PostgresSettings
from qorl.taskset.schemas import Task
from qorl.worker_pool.exceptions import ContainerError

TASK = {
    "task_id": "job-test",
    "template_id": "job-test-template",
    "sql_path": "queries/test.sql",
    "sql_sha256": "unused",
    "tables": ["table_a", "table_b"],
    "table_count": 2,
    "relation_count": 2,
    "join_predicate_count": 1,
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
        def admin_sql(sql: str) -> str:
            del sql
            return ""

        self.task = Task.model_validate(TASK)
        self.max_candidates = 5
        self.sql = "SELECT 1;"
        self.catalog = TaskCatalog.from_task(
            TASK, {"a": {"table_a_pkey"}, "b": {"table_b_pkey"}}
        )
        self._worker = SimpleNamespace(
            settings=PostgresSettings.model_validate(
                {
                    **dict.fromkeys(BOOLEAN_SETTINGS, "on"),
                    **dict.fromkeys(NUMERIC_SETTINGS, "1"),
                    **dict.fromkeys(INTEGER_SETTINGS, "1"),
                }
            ),
            admin_sql=admin_sql,
            fixture=SimpleNamespace(
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


@dataclass
class PlanOnlyWorker:
    settings: PostgresSettings
    indexes: PostgresIndexes
    explain_calls: int = 0

    def explain(
        self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        assert not analyze
        self.explain_calls += 1
        return ExplainResult({"Plan": {"Node Type": "Result"}}, "")

    def admin_sql(self, sql: str) -> str:
        raise AssertionError("no inspection SQL was expected")


class SqlFixture:
    def load_sql(self, task: Task) -> str:
        return "SELECT 1;"


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
            "cca30e47e4b7d6ee393a266c486390a42f0e29f9b5ac295224ea5f0e3a23756a",
            "324507ec86688cd2f91523d7d48d0956ce78c4f39beeff66c27b887e01e0849e",
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
        assert set(trace["initial_observation"]["planner_settings"]) == (
            set(BOOLEAN_SETTINGS) | set(NUMERIC_SETTINGS) | set(INTEGER_SETTINGS)
        )
        assert set(PostgresSettings.model_fields) == set(
            trace["initial_observation"]["planner_settings"]
        )
        assert "postgresql_server_version_num" not in trace["initial_observation"]
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
        worker = PlanOnlyWorker(
            settings=FakeEvaluator(tmp_path).worker.settings,
            indexes=PostgresIndexes(by_table={}),
        )
        evaluator = PlanValidationEvaluator(
            worker,
            SqlFixture(),
            Task.model_validate(TASK),
            default_timeout_ms=5_000,
            max_candidates=1,
        )
        evaluator.start()

        trace = QoAgentPolicy(config(), FakeClient()).search(evaluator)

        assert trace["stop_reason"] == "model_finish"
        assert "candidate-01: validated" in capsys.readouterr().out
        assert worker.explain_calls == 2
        assert (
            evaluator.candidates[0].measurement_status == MeasurementStatus.NOT_MEASURED
        )
        assert evaluator.candidates[0].provisional_speedup is None

    def test_configured_budgets_control_tools_and_enforcement(
        self, tmp_path: Path
    ) -> None:
        client = FakeClient()
        prototype = client.responses[0]
        client.responses = []
        for name, arguments in [
            ("get_plan", '{"candidate_id":"default"}'),
            ("get_plan", '{"candidate_id":"default"}'),
            ("get_plan", '{"candidate_id":"default"}'),
            ("evaluate_candidate", '{"action":{"version":1}}'),
            ("evaluate_candidate", '{"action":{"version":1}}'),
            ("finish", "{}"),
        ]:
            response = deepcopy(prototype)
            response["choices"][0]["message"]["tool_calls"][0]["function"] = {
                "name": name,
                "arguments": arguments,
            }
            client.responses.append(response)
        evaluator = FakeEvaluator(tmp_path)
        evaluator.max_candidates = 1
        settings = AgentSettings(
            candidate_attempts=1, maximum_model_turns=6, inspection_turns_per_alias=1
        )
        trace = QoAgentPolicy(config(), client).search(evaluator, settings=settings)

        assert trace["agent_interface_version"] == 2
        observation = trace["initial_observation"]
        assert observation["candidate_attempts"] == 1
        assert observation["turn_budget"]["total_model_turns"] == 6
        assert observation["turn_budget"]["maximum_inspection_turns"] == 2
        assert trace["stop_reason"] == "model_finish"
        assert len(evaluator.actions) == 1
        events = trace["tool_events"]
        assert "Plan" in events[0]["result"] and "Plan" in events[1]["result"]
        for index in (2, 4):
            assert (
                events[index]["result"]["error"]
                == "tool is not available for this turn"
            )
        assert {tool["function"]["name"] for tool in client.requests[2]["tools"]} == {
            "evaluate_candidate",
            "keep_default",
        }
        assert {tool["function"]["name"] for tool in client.requests[4]["tools"]} == {
            "finish"
        }

    def test_configured_model_turn_limit_stops_the_loop(self, tmp_path: Path) -> None:
        client = FakeClient()
        evaluator = FakeEvaluator(tmp_path)
        evaluator.max_candidates = 1
        trace = QoAgentPolicy(config(), client).search(
            evaluator,
            settings=AgentSettings(
                candidate_attempts=1,
                maximum_model_turns=1,
                inspection_turns_per_alias=0,
            ),
        )
        assert trace["stop_reason"] == "model_turn_limit"
        assert len(client.requests) == 1

    def test_inconsistent_candidate_limits_fail_before_model_request(
        self, tmp_path: Path
    ) -> None:
        client = FakeClient()
        evaluator = FakeEvaluator(tmp_path)
        with pytest.raises(ValueError, match="candidate limits must agree"):
            QoAgentPolicy(config(), client).search(
                evaluator,
                settings=AgentSettings(
                    candidate_attempts=1,
                    maximum_model_turns=6,
                    inspection_turns_per_alias=1,
                ),
            )
        assert client.requests == []

    def test_keep_default_ends_without_a_candidate(self, tmp_path: Path) -> None:
        client = KeepDefaultClient()
        policy = QoAgentPolicy(config(), client)
        evaluator = FakeEvaluator(tmp_path)

        trace = policy.search(evaluator)

        assert trace["stop_reason"] == "model_keep_default"
        assert evaluator.kept_default
        assert evaluator.candidates == []
        assert trace["tool_events"][0]["result"]["status"] == "kept_default"

    @pytest.mark.parametrize(
        "error",
        [PostgresError("connection lost"), ContainerError("container disappeared")],
    )
    def test_plan_validation_infrastructure_failure_escapes_the_agent_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: PostgresError | ContainerError,
    ) -> None:
        worker = PlanOnlyWorker(
            settings=FakeEvaluator(tmp_path).worker.settings,
            indexes=PostgresIndexes(by_table={}),
        )
        evaluator = PlanValidationEvaluator(
            worker,
            SqlFixture(),
            Task.model_validate(TASK),
            default_timeout_ms=5_000,
            max_candidates=1,
        )
        evaluator.start()

        def fail(
            sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
        ) -> ExplainResult:
            raise error

        monkeypatch.setattr(worker, "explain", fail)
        client = FakeClient()
        with pytest.raises(type(error), match=str(error)):
            QoAgentPolicy(config(), client).search(evaluator)
        assert len(client.requests) == 1
        assert evaluator.candidates == []

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
