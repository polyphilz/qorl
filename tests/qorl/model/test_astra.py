"""Astra wire format, encrypted continuation, and full-history budget checks."""

import copy
import os
import tomllib
from collections import deque
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from qorl.agent.tools import agent_tools
from qorl.model.client import AstraModelClient
from qorl.model.exceptions import ContextBudgetError, ModelError
from qorl.model.schemas import (
    AstraInferenceSettings,
    GenerationRequest,
    JsonObject,
    Message,
    MessageRole,
    ModelPreset,
    ModelSettings,
    ToolDefinition,
)

PROMPT_TOKENS = 100
ARGUMENTS = '{ "action": { "version": 1 } }'


class ScriptedTransport:
    def __init__(self, responses: list[JsonObject]) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[str, JsonObject | None]] = []

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
        self.requests.append((path, copy.deepcopy(body)))
        return self.responses.popleft()


@pytest.fixture
def preset(repository_root: Path) -> ModelPreset:
    path = repository_root / "configs/defaults/models/000-gpt-6-astra.toml"
    return ModelPreset.model_validate(tomllib.loads(path.read_text()))


@pytest.fixture
def turn() -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="Use the tools."),
            Message(role=MessageRole.USER, content="Optimize this query."),
        ],
        tools=[ToolDefinition.model_validate(tool) for tool in agent_tools(["a", "b"])],
        seed=42,
    )


def reply() -> JsonObject:
    return {
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "opaque-token-state",
                "summary": [],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "commentary",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will propose a plan.",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "evaluate_candidate",
                "arguments": ARGUMENTS,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    }


def astra(preset: ModelPreset, transport: ScriptedTransport) -> AstraModelClient:
    assert isinstance(preset.inference, AstraInferenceSettings)
    return AstraModelClient(preset.model, preset.inference, transport=transport)


def test_preserves_full_responses_output_and_argument_bytes(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    counted: JsonObject = {"input_tokens": PROMPT_TOKENS}
    transport = ScriptedTransport([counted, reply(), counted, reply()])
    model = astra(preset, transport)
    first = model.generate(turn)
    assert first.message.tool_calls is not None
    assert first.message.tool_calls[0].id == "call_1"
    assert first.message.tool_calls[0].function.arguments == ARGUMENTS
    assert first.usage.reasoning_tokens == 30
    assert first.usage.cached_tokens == 20
    continued = turn.model_copy(
        update={
            "messages": [
                *turn.messages,
                first.message,
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id="call_1",
                    name="evaluate_candidate",
                    content='{"candidate_id":"candidate-01"}',
                ),
            ]
        }
    )
    second = model.generate(continued)
    inputs = second.request["input"]
    assert isinstance(inputs, list)
    assert inputs[2:-1] == reply()["output"]
    assert inputs[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"candidate_id":"candidate-01"}',
    }
    assert second.raw_response == reply()
    assert second.request["store"] is False
    assert second.request["truncation"] == "disabled"
    assert second.request["parallel_tool_calls"] is False
    assert second.request["tool_choice"] == "required"
    assert second.request["reasoning"] == {"effort": "medium"}
    for field in (
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "chat_template_kwargs",
        "previous_response_id",
    ):
        assert field not in second.request
    count_body = transport.requests[2][1]
    assert count_body == {
        key: value
        for key, value in second.request.items()
        if key not in {"store", "max_output_tokens"}
    }
    tools = second.request["tools"]
    assert isinstance(tools, list)
    for tool, definition in zip(tools, turn.tools, strict=True):
        assert isinstance(tool, dict)
        assert tool["parameters"] == definition.function.parameters
        assert tool["strict"] is False
    assert [path for path, _ in transport.requests] == [
        "responses/input_tokens",
        "responses",
        "responses/input_tokens",
        "responses",
    ]


@pytest.mark.parametrize("excess", [0, 1])
def test_counts_before_generation_and_never_trims(
    preset: ModelPreset, turn: GenerationRequest, excess: int
) -> None:
    count = preset.model.context_length - preset.inference.max_tokens + excess
    transport = ScriptedTransport([{"input_tokens": count}, reply()])
    if excess:
        with pytest.raises(ContextBudgetError):
            astra(preset, transport).generate(turn)
        assert len(transport.requests) == 1
    else:
        assert astra(preset, transport).generate(turn).prompt_tokens == count


def test_truncation_is_not_a_content_retry(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    response: JsonObject = {
        **reply(),
        "status": "incomplete",
        "output": [],
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    transport = ScriptedTransport([{"input_tokens": PROMPT_TOKENS}, response])
    result = astra(preset, transport).generate(turn)
    assert result.truncated
    assert result.finish_reason == "max_output_tokens"
    assert result.message.tool_calls is None
    assert result.raw_response == response
    assert len(transport.requests) == 2


def test_text_only_response_is_not_retried(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    response = {**reply(), "output": []}
    transport = ScriptedTransport([{"input_tokens": PROMPT_TOKENS}, response])
    assert astra(preset, transport).generate(turn).message.tool_calls is None
    assert not transport.responses


def test_refuses_reconstructed_assistant_history(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    turn = turn.model_copy(
        update={
            "messages": [
                *turn.messages,
                Message(role=MessageRole.ASSISTANT, content="lost reasoning"),
            ]
        }
    )
    transport = ScriptedTransport([])
    with pytest.raises(ModelError, match="original Responses items"):
        astra(preset, transport).generate(turn)
    assert not transport.requests


@pytest.mark.parametrize("status", ["failed", "in_progress", "cancelled"])
def test_provider_failures_are_not_policy_decisions(
    preset: ModelPreset, turn: GenerationRequest, status: str
) -> None:
    transport = ScriptedTransport(
        [{"input_tokens": PROMPT_TOKENS}, {**reply(), "status": status}]
    )
    with pytest.raises(ModelError, match="did not complete"):
        astra(preset, transport).generate(turn)


@pytest.mark.parametrize(
    "field,value",
    [("temperature", 1), ("thinking", False), ("reasoning_effort", "none")],
)
def test_unsupported_inference_settings_are_rejected(
    preset: ModelPreset, field: str, value: str | int | bool
) -> None:
    with pytest.raises(ValidationError):
        AstraInferenceSettings.model_validate(
            {**preset.inference.model_dump(), field: value}
        )


def test_fable_provider_is_not_selectable() -> None:
    with pytest.raises(ValidationError):
        ModelSettings.model_validate(
            {
                "provider": "anthropic",
                "name_or_path": "claude-fable-5-1",
                "context_length": 1,
            }
        )


@pytest.mark.skipif(
    os.environ.get("QORL_TEST_ASTRA") != "1",
    reason="live Astra gate requires QORL_TEST_ASTRA=1 and OPENAI_API_KEY",
)
def test_live_tool_continuation(
    preset: ModelPreset,
    record_property: Callable[[str, str], None],
) -> None:
    """Opt-in paid API check: inspect a synthetic plan, then keep it unchanged."""
    assert isinstance(preset.inference, AstraInferenceSettings)
    client = AstraModelClient(preset.model, preset.inference)
    tools = [ToolDefinition.model_validate(tool) for tool in agent_tools(["a", "b"])]
    initial = GenerationRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="Call get_plan for default, then keep_default. This is a tool-transport test.",
            ),
            Message(role=MessageRole.USER, content="Inspect the default plan."),
        ],
        tools=[tool for tool in tools if tool.function.name == "get_plan"],
    )
    first = client.generate(initial)
    record_property("first", first.model_dump_json())
    assert not first.truncated
    assert first.message.tool_calls is not None and len(first.message.tool_calls) == 1
    call = first.message.tool_calls[0]
    assert call.function.name == "get_plan"
    assert first.message.continuation is not None
    second = client.generate(
        GenerationRequest(
            messages=[
                *initial.messages,
                first.message,
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id=call.id,
                    name="get_plan",
                    content='{"Plan":{"Node Type":"Result"}}',
                ),
            ],
            tools=[tool for tool in tools if tool.function.name == "keep_default"],
        )
    )
    record_property("second", second.model_dump_json())
    assert not second.truncated
    assert second.message.tool_calls is not None and len(second.message.tool_calls) == 1
    assert second.message.tool_calls[0].function.name == "keep_default"
