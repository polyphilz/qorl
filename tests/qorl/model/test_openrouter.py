"""OpenRouter wire evidence and post-generation budget enforcement, without HTTP."""

import copy
import io
import json
import urllib.error
import urllib.request
from http.client import HTTPMessage

import pytest
from pydantic import ValidationError
from tests.qorl.model.test_astra import ScriptedTransport
from tests.qorl.model.test_astra import turn as turn

from qorl.model import client as clients
from qorl.model.client import AstraModelClient, OpenRouterModelClient, model_client
from qorl.model.exceptions import ModelError, ModelResponseError
from qorl.model.schemas import (
    OPENROUTER_MODEL_ID,
    AstraInferenceSettings,
    GenerationRequest,
    JsonObject,
    Message,
    MessageRole,
    ModelPreset,
    ModelProvider,
    OpenRouterContinuation,
    OpenRouterInferenceSettings,
    ReasoningEffort,
    TokenUsage,
)

ARGUMENTS = '{ "action": {"version": 1} }'
DETAILS: list[JsonObject] = [
    {"type": "reasoning.text", "text": "Inspect before proposing.", "index": 0},
    {
        "type": "reasoning.encrypted",
        "data": "OPAQUE",
        "format": "unknown-v1",
        "index": 1,
    },
]


@pytest.fixture
def preset(openrouter_preset: ModelPreset) -> ModelPreset:
    return openrouter_preset


def reply() -> JsonObject:
    return {
        "model": OPENROUTER_MODEL_ID,
        "provider": "example-backend",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Inspect before proposing.",
                    "reasoning_details": copy.deepcopy([*DETAILS]),
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "evaluate_candidate",
                                "arguments": ARGUMENTS,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 30},
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    }


def client(preset: ModelPreset, transport: ScriptedTransport) -> OpenRouterModelClient:
    assert isinstance(preset.inference, OpenRouterInferenceSettings)
    return OpenRouterModelClient(preset.model, preset.inference, transport=transport)


def test_current_tools_and_exact_continuation(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    raw = reply()
    transport = ScriptedTransport([raw])
    connection = client(preset, transport)
    response = connection.generate(turn)
    assert transport.requests == [("chat/completions", response.request)]
    assert response.raw_response == raw
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].function.arguments == ARGUMENTS
    assert response.prompt_tokens == response.usage.prompt_tokens == 100
    assert response.usage.reasoning_tokens == 30
    assert response.usage.cached_tokens == 20
    body = response.request
    assert body["reasoning"] == {"effort": "medium", "exclude": False}
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert body["provider"] == {"require_parameters": True}
    assert body["seed"] == 42
    assert body["top_k"] == 20 and body["top_p"] == 0.95
    assert body["tool_choice"] == "auto" and "parallel_tool_calls" not in body
    following = GenerationRequest(
        messages=[
            *turn.messages,
            response.message,
            Message(
                role=MessageRole.TOOL, content="Actual feedback", tool_call_id="call-1"
            ),
        ],
        tools=[turn.tools[-1]],
        seed=43,
    )
    next_body = connection.request_body(following)
    assert next_body["tools"] == [turn.tools[-1].model_dump(mode="json")]
    messages = next_body["messages"]
    assert isinstance(messages, list)
    assistant = messages[-2]
    assert isinstance(assistant, dict)
    assert assistant["reasoning_details"] == DETAILS
    assert "reasoning" not in assistant and "reasoning_content" not in assistant
    assert (
        Message.model_validate_json(response.message.model_dump_json())
        == response.message
    )


@pytest.mark.parametrize("details", [None, [], DETAILS])
def test_text_fallback_and_empty_reasoning(
    preset: ModelPreset, turn: GenerationRequest, details: list[JsonObject] | None
) -> None:
    raw = reply()
    choices = raw["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    message = choices[0]["message"]
    assert isinstance(message, dict)
    message["reasoning"] = ""
    message["reasoning_details"] = [*details] if details is not None else None
    response = client(preset, ScriptedTransport([raw])).generate(turn)
    assert response.message.reasoning_content == (
        "Inspect before proposing." if details else None
    )
    assert response.message.continuation == OpenRouterContinuation(
        model=OPENROUTER_MODEL_ID, reasoning_details=details
    )


@pytest.mark.parametrize(
    "change",
    [
        {"usage": None},
        {"usage": {"completion_tokens": 3}},
        {"usage": {"prompt_tokens": 999999}},
        {"choices": []},
        {"model": "qwen/max-snapshot"},
        {"error": {"message": "failed"}},
    ],
)
def test_failed_replies_retain_raw_evidence(
    preset: ModelPreset, turn: GenerationRequest, change: JsonObject
) -> None:
    raw = {**reply(), **change}
    transport = ScriptedTransport([raw])
    with pytest.raises(ModelResponseError) as caught:
        client(preset, transport).generate(turn)
    assert caught.value.evidence.raw_response == raw
    assert caught.value.evidence.request == transport.requests[0][1]
    assert len(transport.requests) == 1


@pytest.mark.parametrize("reason", ["length", "content_filter", "error", "unknown"])
def test_finish_reasons(
    preset: ModelPreset, turn: GenerationRequest, reason: str
) -> None:
    raw = reply()
    choices = raw["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    choices[0]["finish_reason"] = reason
    connection = client(preset, ScriptedTransport([raw]))
    if reason == "length":
        response = connection.generate(turn)
        assert response.truncated and response.message.tool_calls
    else:
        with pytest.raises(ModelResponseError):
            connection.generate(turn)


@pytest.mark.parametrize("reason", ["length", "stop", "tool_calls"])
def test_empty_length_reply_is_a_truncation(
    preset: ModelPreset, turn: GenerationRequest, reason: str
) -> None:
    raw: JsonObject = {
        **reply(),
        "choices": [
            {"message": {"role": "assistant", "content": None}, "finish_reason": reason}
        ],
    }
    transport = ScriptedTransport([raw])
    if reason == "length":
        response = client(preset, transport).generate(turn)
        assert response.truncated and response.finish_reason == "length"
        assert response.message.content is None
        assert response.message.reasoning_content is None
        assert not response.message.tool_calls
        assert response.usage.completion_tokens == 50
    else:
        with pytest.raises(ModelResponseError, match="empty assistant"):
            client(preset, transport).generate(turn)
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"choices": [{"message": {"role": "user"}, "finish_reason": "length"}]},
        {"choices": [{"message": None, "finish_reason": "length"}]},
        {"usage": {"prompt_tokens": 999999}},
    ],
)
def test_empty_length_reply_still_checks_context_and_schema(
    preset: ModelPreset, turn: GenerationRequest, change: JsonObject
) -> None:
    raw: JsonObject = {
        **reply(),
        "choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}],
        **change,
    }
    with pytest.raises(ModelResponseError):
        client(preset, ScriptedTransport([raw])).generate(turn)


@pytest.mark.parametrize(
    "invalid_usage",
    [
        None,
        {},
        {"prompt_tokens": -1, "completion_tokens": 50},
        {"prompt_tokens": "100", "completion_tokens": 50},
        {"prompt_tokens": True, "completion_tokens": 50},
        {"prompt_tokens": 100, "completion_tokens_details": {"reasoning_tokens": -1}},
    ],
)
def test_invalid_or_missing_usage_is_not_recovered_per_field(
    preset: ModelPreset, turn: GenerationRequest, invalid_usage: JsonObject | None
) -> None:
    raw: JsonObject = {**reply(), "choices": [], "usage": invalid_usage}
    with pytest.raises(ModelResponseError) as caught:
        client(preset, ScriptedTransport([raw])).generate(turn)
    assert caught.value.evidence.usage == TokenUsage()
    assert caught.value.evidence.raw_response == raw


def test_malformed_choices_keep_independently_validated_usage(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    raw = {**reply(), "choices": []}
    with pytest.raises(ModelResponseError) as caught:
        client(preset, ScriptedTransport([raw])).generate(turn)
    assert caught.value.evidence.usage == TokenUsage(
        prompt_tokens=100, completion_tokens=50, reasoning_tokens=30, cached_tokens=20
    )


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": 131073},
        {"max_tokens": 0},
        {"thinking": False},
        {"reasoning_effort": "high"},
        {"reasoning_effort": "none"},
    ],
)
def test_inference_rejections(preset: ModelPreset, change: JsonObject) -> None:
    with pytest.raises(ValidationError):
        OpenRouterInferenceSettings.model_validate(
            {**preset.inference.model_dump(), **change}
        )


def test_factory_and_continuation_guards(
    preset: ModelPreset, turn: GenerationRequest
) -> None:
    assert isinstance(
        model_client(preset.model, preset.inference), OpenRouterModelClient
    )
    response = client(preset, ScriptedTransport([reply()])).generate(turn)
    history = GenerationRequest(messages=[response.message], tools=[])
    wrong = response.message.model_copy(
        update={
            "continuation": OpenRouterContinuation(
                model="another-model", reasoning_details=DETAILS
            )
        }
    )
    with pytest.raises(ModelError, match="another provider, model or role"):
        client(preset, ScriptedTransport([])).request_body(
            history.model_copy(update={"messages": [wrong]})
        )
    astra = AstraModelClient(
        preset.model.model_copy(
            update={"provider": ModelProvider.OPENAI, "name_or_path": "gpt-6-astra"}
        ),
        AstraInferenceSettings(max_tokens=100, reasoning_effort=ReasoningEffort.MEDIUM),
        transport=ScriptedTransport([]),
    )
    with pytest.raises(ModelError, match="cannot be sent to Astra"):
        astra.request_body(history)


def test_transport_credentials_and_retry_keep_request_identity(
    preset: ModelPreset,
    turn: GenerationRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def opened(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        requests.append(request)
        assert timeout == 600
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "busy", HTTPMessage(), io.BytesIO(b"retry")
            )
        return io.BytesIO(json.dumps(reply()).encode())

    monkeypatch.setattr(clients.urllib.request, "urlopen", opened)

    def no_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(clients.time, "sleep", no_sleep)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    connection = model_client(preset.model, preset.inference)
    with pytest.raises(ModelError, match="OPENROUTER_API_KEY"):
        connection.generate(turn)
    assert not requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    response = connection.generate(turn)
    assert len(requests) == 2
    assert requests[0].data == requests[1].data
    assert requests[0].get_header("Idempotency-key") == requests[1].get_header(
        "Idempotency-key"
    )
    assert requests[0].get_header("Authorization") == "Bearer mock-key"
    assert requests[0].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert "mock-key" not in response.model_dump_json()
