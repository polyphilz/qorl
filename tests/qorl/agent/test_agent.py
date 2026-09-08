"""Exercise the shared conversation loop with real clients and plan validation."""

import hashlib
import json
import tomllib
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from qorl.agent.agent import QoAgentPolicy, turn_seed
from qorl.agent.interface import AGENT_INTERFACE_VERSION, AgentInterface
from qorl.agent.schemas import AgentSettings, AgentTrace
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.types import InspectionExecutor, StopReason
from qorl.inference.schemas import ServingSettings
from qorl.measure.schemas import MeasurementStatus
from qorl.measure.validation import PlanValidationEvaluator
from qorl.model.client import JSON_OBJECT, AstraModelClient, LocalModelClient
from qorl.model.exceptions import ModelError
from qorl.model.schemas import (
    JsonObject,
    LocalInferenceSettings,
    MessageRole,
    ModelPreset,
    ModelProvider,
    ModelSettings,
)
from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import (
    ExplainResult,
    PostgresIndexes,
    PostgresSettings,
    WorkerAllocation,
)
from qorl.taskset.schemas import Task
from qorl.worker_pool.exceptions import ContainerError

CONTEXT_LENGTH = 20_480
SERVING = ServingSettings(
    host="127.0.0.1",
    port=8000,
    dtype="bfloat16",
    tool_call_parser="qwen3_coder",
    reasoning_parser="qwen3",
    max_num_seqs=4,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
    use_flashinfer_sampler=False,
    startup_timeout_seconds=600,
)
OUTPUT_TOKENS = 2_048
PROMPT_TOKENS = 100
MODEL_TURNS = 64
SEED = 7
TIMEOUT_MS = 5_000

TASK = Task.model_validate(
    {
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
)


@dataclass
class Database:
    settings: PostgresSettings
    indexes: PostgresIndexes
    explain_calls: int = 0
    allocation: WorkerAllocation | None = None

    def explain(
        self, sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        assert not analyze
        self.explain_calls += 1
        return ExplainResult(
            {"Plan": {"Node Type": "Result", "Plan Rows": 1, "Total Cost": 99.0}}, ""
        )

    def admin_sql(self, sql: str) -> str:
        raise AssertionError(
            "this test inspects the stored plan, not database metadata"
        )


class SqlFixture:
    def load_sql(self, task: Task) -> str:
        return "SELECT 1;"


@pytest.fixture
def database(repository_root: Path) -> Database:
    from qorl.postgres.config import PostgresConfig

    pgconf = PostgresConfig.load(
        repository_root / "docker/postgres/configs/000-pgconf-default"
    )
    return Database(pgconf.agent_settings, PostgresIndexes(by_table={}))


@pytest.fixture
def evaluator(database: Database) -> PlanValidationEvaluator[InspectionExecutor]:
    evaluator = PlanValidationEvaluator[InspectionExecutor](
        database,
        SqlFixture(),
        TASK,
        default_timeout_ms=TIMEOUT_MS,
        max_candidates=1,
    )
    evaluator.start()
    return evaluator


def reply(name: str, arguments: str = "{}") -> JsonObject:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "choose a tool",
                    "tool_calls": [
                        {
                            "id": name,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": 1},
    }


class ScriptedTransport:
    def __init__(self, responses: list[JsonObject]) -> None:
        self.responses = responses
        self.requests: list[JsonObject] = []
        self.token_requests: list[JsonObject] = []
        self.count = PROMPT_TOKENS

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
        assert body is not None
        if path in ("../tokenize", "responses/input_tokens"):
            self.token_requests.append(body)
            return {
                "count": self.count,
                "max_model_len": CONTEXT_LENGTH,
                "input_tokens": self.count,
            }
        assert path in ("chat/completions", "responses")
        self.requests.append(body)
        return self.responses.pop(0)


def policy(
    transport: ScriptedTransport,
    *,
    turns: int = MODEL_TURNS,
    inspections: int = 3,
    attempts: int = 1,
    seed: int | None = SEED,
) -> QoAgentPolicy:
    model = ModelSettings(
        provider=ModelProvider.LOCAL,
        name_or_path="test-model",
        context_length=CONTEXT_LENGTH,
        base_url="http://example.test/v1",
        request_timeout_seconds=1,
    )
    inference = LocalInferenceSettings(
        max_tokens=OUTPUT_TOKENS,
        temperature=1,
        top_p=1,
        top_k=0,
        min_p=0,
        presence_penalty=0,
        repetition_penalty=1,
        thinking=False,
        serving=SERVING,
    )
    return QoAgentPolicy(
        LocalModelClient(model, inference, transport=transport),
        AgentSettings(
            candidate_attempts=attempts,
            maximum_model_turns=turns,
            inspection_turns_per_alias=inspections,
        ),
        context_length=CONTEXT_LENGTH,
        max_tokens=OUTPUT_TOKENS,
        seed=seed,
    )


def test_request_goldens(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}'), reply("finish")]
    )
    policy(transport).search(evaluator)
    # Final combined interface-v6 request bytes, including the committed prompt line break.
    assert [
        hashlib.sha256(json.dumps(request).encode()).hexdigest()
        for request in transport.requests
    ] == [
        "aa47d42a190e1b43412f91188e5cef0db60e35c111e7c684cb6cba86436226d3",
        "95e97a5f9e5a41e1d65ccbd230b4b4baeb6f3d46bd9e270278ea0ab4f129990c",
    ]


def test_evaluate_then_finish(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
    database: Database,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}'), reply("finish")]
    )
    trace = policy(transport).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_FINISH
    assert trace.agent_interface_version == AGENT_INTERFACE_VERSION
    assert trace.usage.prompt_tokens == PROMPT_TOKENS * 2
    assert trace.usage.reasoning_tokens is None
    assert trace.prompt_tokens == PROMPT_TOKENS
    assert database.explain_calls == 2
    assert "candidate-01: validated" in capsys.readouterr().out
    assert evaluator.candidates[0].measurement_status == MeasurementStatus.NOT_MEASURED
    assert [message.role for message in trace.transcript] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    feedback = trace.tool_events[0].result
    assert feedback["action_valid"] and feedback["constraints_satisfied"]
    assert feedback["structurally_novel"] is False
    assert feedback["structural_duplicate_of"] == "default"
    assert feedback["attempts_remaining"] == 0
    for field in ("plan_sha256", "structural_plan_sha256", "timing_reuse_key"):
        assert field not in feedback
        assert getattr(evaluator.candidates[0], field) is not None
    assert AgentTrace.model_validate_json(trace.model_dump_json()) == trace
    for message in trace.transcript:
        if message.role == MessageRole.TOOL:
            assert message.content is not None
            event = next(
                item
                for item in trace.tool_events
                if item.tool_call_id == message.tool_call_id
            )
            assert message.content == json.dumps(event.result, sort_keys=True)
    initial = trace.initial_observation.planner_settings
    limits = trace.initial_observation.resource_limits.postgres
    assert set(type(initial).model_fields) | set(type(limits).model_fields) == set(
        PostgresSettings.model_fields
    )
    assert (
        "postgresql_server_version_num"
        not in type(trace.initial_observation).model_fields
    )
    assert transport.requests[1]["messages"] == transport.token_requests[1]["messages"]
    assert transport.requests[1]["messages"] != transport.requests[0]["messages"]
    assert '"reasoning": "choose a tool"' in json.dumps(transport.requests[1])


def test_budgets_mask_tools_and_reject_unavailable_calls(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    transport = ScriptedTransport(
        [
            reply("get_plan", '{"candidate_id":"default"}'),
            reply("get_plan", '{"candidate_id":"default"}'),
            reply("get_plan", '{"candidate_id":"default"}'),
            reply("evaluate_candidate", '{"action":{"version":1}}'),
            reply("evaluate_candidate", '{"action":{"version":1}}'),
            reply("finish"),
        ]
    )
    trace = policy(transport, turns=6, inspections=1).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_FINISH
    assert len(evaluator.candidates) == 1
    for index in (0, 1):
        nodes = trace.tool_events[index].result["nodes"]
        assert isinstance(nodes, list)
        assert JSON_OBJECT.validate_python(nodes[0])["estimates"] == {
            "Node Type": "Result",
            "Plan Rows": 1,
            "Total Cost": 99.0,
        }
    for index in (2, 4):
        assert (
            trace.tool_events[index].result["error"]
            == "tool is not available for this turn"
        )
    tools = transport.requests[4]["tools"]
    assert isinstance(tools, list)
    assert [JSON_OBJECT.validate_python(tool)["function"] for tool in tools] == [
        next(
            tool.function.model_dump(mode="json")
            for tool in trace.tools
            if tool.function.name == "finish"
        )
    ]


def test_context_budget_is_counted_before_generation(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    transport = ScriptedTransport([])
    transport.count = CONTEXT_LENGTH - OUTPUT_TOKENS + 1
    trace = policy(transport).search(evaluator)
    assert trace.stop_reason == StopReason.CONTEXT_BUDGET
    assert trace.prompt_tokens == transport.count
    assert len(transport.token_requests) == 1
    assert transport.requests == []
    assert evaluator.candidates == []


@pytest.mark.parametrize("arguments", ["not-json", "[]", '{"action":{"unknown":1}}'])
def test_malformed_decision_consumes_a_slot_without_sql(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
    database: Database,
    arguments: str,
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", arguments), reply("finish")]
    )
    trace = policy(transport).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_FINISH
    assert len(evaluator.candidates) == 1
    assert not evaluator.candidates[0].action_valid
    assert database.explain_calls == 1
    assert trace.tool_events[0].result["attempts_remaining"] == 0


def test_two_candidates_can_use_validation_feedback(database: Database) -> None:
    evaluator = PlanValidationEvaluator[InspectionExecutor](
        database,
        SqlFixture(),
        TASK,
        default_timeout_ms=TIMEOUT_MS,
        max_candidates=2,
    )
    evaluator.start()
    transport = ScriptedTransport(
        [
            reply("evaluate_candidate", "not-json"),
            reply("evaluate_candidate", '{"action":{"version":1}}'),
            reply("finish"),
        ]
    )
    trace = policy(transport, attempts=2).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_FINISH
    assert [candidate.action_valid for candidate in evaluator.candidates] == [
        False,
        True,
    ]
    assert database.explain_calls == 2
    assert trace.tool_events[0].result["errors_or_diagnostics"]
    assert trace.tool_events[0].result["attempts_remaining"] == 1


def test_missing_usage_does_not_block_counted_turns(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    first = reply("evaluate_candidate", '{"action":{"version":1}}')
    first.pop("usage")
    transport = ScriptedTransport([first, reply("finish")])
    trace = policy(transport).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_FINISH
    assert trace.usage.prompt_tokens is None
    assert len(transport.token_requests) == 2


def test_truncated_tool_is_never_executed(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    truncated = reply("keep_default")
    choices = truncated["choices"]
    assert isinstance(choices, list)
    choice = JSON_OBJECT.validate_python(choices[0])
    choice["finish_reason"] = "length"
    truncated["choices"] = [choice]
    trace = policy(ScriptedTransport([truncated])).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_OUTPUT_LIMIT
    assert not evaluator.kept_default
    assert trace.tool_events == []
    assert trace.model_responses[0].truncated
    assert trace.transcript[-1].tool_calls is not None


def test_turn_limit_and_candidate_limit(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}')]
    )
    with pytest.raises(ValueError, match="candidate limits must agree"):
        policy(transport, attempts=2).search(evaluator)
    assert transport.requests == []
    assert (
        policy(transport, turns=1).search(evaluator).stop_reason
        == StopReason.MODEL_TURN_LIMIT
    )
    assert len(transport.requests) == 1


def test_terminal_order(evaluator: PlanValidationEvaluator[InspectionExecutor]) -> None:
    environment = AgentEnvironment(evaluator)
    result, finished = environment.execute("finish", {})
    assert isinstance(result["error"], str)
    assert "use keep_default" in result["error"]
    assert not finished
    evaluator.evaluate({"version": 1})
    result, finished = environment.execute("keep_default", {})
    assert isinstance(result["error"], str)
    assert "before submitting a candidate" in result["error"]
    assert not finished
    result, finished = environment.execute(
        "evaluate_candidate", {"action": {"version": 1}}
    )
    assert result == {"error": "rollout candidate budget is exhausted"}
    assert not finished


def test_keep_default_and_unseeded_requests(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    transport = ScriptedTransport([reply("keep_default")])
    trace = policy(transport, seed=None).search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_KEEP_DEFAULT
    assert evaluator.kept_default and not evaluator.candidates
    assert "seed" not in transport.requests[0]
    assert turn_seed(SEED, TASK.task_id, 1) == turn_seed(SEED, TASK.task_id, 1)
    assert turn_seed(SEED, TASK.task_id, 1) != turn_seed(SEED, TASK.task_id, 2)


@pytest.mark.parametrize("error", [PostgresError("lost"), ContainerError("lost")])
def test_failure_keeps_submitted_evidence_unfinished(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}')]
    )
    agent = policy(transport)

    def fail(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        raise error

    monkeypatch.setattr(evaluator.worker, "explain", fail)
    with pytest.raises(type(error), match="lost"):
        agent.search(evaluator)
    assert agent.trace is not None
    assert agent.trace.stop_reason is None
    assert len(agent.trace.model_responses) == 1
    assert agent.trace.transcript[-1].tool_calls is not None
    assert evaluator.candidates == []


def test_model_failure_retains_completed_tool_evidence(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}')]
    )
    original = transport.request

    def fail_on_second_turn(path: str, body: JsonObject | None = None) -> JsonObject:
        if path == "chat/completions" and transport.requests:
            raise ModelError("provider unavailable")
        return original(path, body)

    monkeypatch.setattr(transport, "request", fail_on_second_turn)
    agent = policy(transport)
    with pytest.raises(ModelError, match="provider unavailable"):
        agent.search(evaluator)
    assert agent.trace is not None and agent.trace.stop_reason is None
    assert len(agent.trace.tool_events) == 1
    assert len(evaluator.candidates) == 1


def test_cancel_after_validation_retains_candidate(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = Event()
    evaluator.cancel = cancel
    original = evaluator.worker.explain

    def complete_then_cancel(
        sql: str, timeout_ms: int, *, analyze: bool = False, hint: str = ""
    ) -> ExplainResult:
        result = original(sql, timeout_ms, analyze=analyze, hint=hint)
        cancel.set()
        return result

    monkeypatch.setattr(evaluator.worker, "explain", complete_then_cancel)
    agent = policy(
        ScriptedTransport([reply("evaluate_candidate", '{"action":{"version":1}}')])
    )
    with pytest.raises(CancelledError):
        agent.search(evaluator)
    assert len(evaluator.candidates) == 1
    assert agent.trace is not None and agent.trace.stop_reason is None


def test_astra_continuation_survives_real_tool_loop(
    repository_root: Path,
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    from qorl.model.schemas import AstraInferenceSettings

    preset = ModelPreset.model_validate(
        tomllib.loads(
            (
                repository_root / "configs/defaults/models/000-gpt-6-astra.toml"
            ).read_text()
        )
    )
    assert isinstance(preset.inference, AstraInferenceSettings)
    output: list[JsonObject] = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "opaque",
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Inspect the default."}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_plan",
            "arguments": '{"candidate_id":"default"}',
        },
    ]
    first: JsonObject = {
        "model": "gpt-6-astra",
        "status": "completed",
        "output": list(output),
    }
    second: JsonObject = {
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "keep_default",
                "arguments": "{}",
            }
        ],
    }
    transport = ScriptedTransport([first, second])
    agent = QoAgentPolicy(
        AstraModelClient(preset.model, preset.inference, transport=transport),
        AgentSettings(
            candidate_attempts=1,
            maximum_model_turns=MODEL_TURNS,
            inspection_turns_per_alias=3,
        ),
        context_length=preset.model.context_length,
        max_tokens=preset.inference.max_tokens,
        seed=SEED,
    )
    trace = agent.search(evaluator)
    assert trace.stop_reason == StopReason.MODEL_KEEP_DEFAULT
    inputs = transport.requests[1]["input"]
    assert isinstance(inputs, list)
    assert inputs[2:5] == output
    assert inputs[5] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": trace.transcript[3].content,
    }
    assert trace.transcript[2].continuation is not None
    assert trace.transcript[2].continuation.output == output
    assert trace.model_responses[0].raw_response == first
    assert "seed" not in transport.requests[0]
    assert transport.token_requests[1]["input"] == inputs


def test_single_candidate_observation_and_recursive_schema(
    evaluator: PlanValidationEvaluator[InspectionExecutor],
) -> None:
    interface = AgentInterface.from_evaluator(evaluator, MODEL_TURNS)
    assert interface.observation.candidate_attempts == 1
    assert interface.observation.turn_budget.reserved_final_turns == 2
    assert interface.available_tool_names(1, 1) == {"finish"}
    tool = next(
        item for item in interface.tools if item.function.name == "evaluate_candidate"
    )
    schema = tool.function.parameters
    properties = JSON_OBJECT.validate_python(schema["properties"])
    action = JSON_OBJECT.validate_python(properties["action"])
    action_properties = JSON_OBJECT.validate_python(action["properties"])
    assert (
        JSON_OBJECT.validate_python(action_properties["leading"])["$ref"]
        == "#/$defs/JoinNode"
    )
    definitions = JSON_OBJECT.validate_python(schema["$defs"])
    join = JSON_OBJECT.validate_python(definitions["JoinNode"])
    join_properties = JSON_OBJECT.validate_python(join["properties"])
    left = JSON_OBJECT.validate_python(join_properties["left"])
    alternatives = left["anyOf"]
    assert isinstance(alternatives, list)
    assert JSON_OBJECT.validate_python(alternatives[0])["enum"] == ["a", "b"]
