"""Local model wire-format, continuation, identity, and context-budget checks."""

import copy
import io
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from http.client import HTTPMessage
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from verifiers.v1.clients import train as proxy_train
from verifiers.v1.types import AssistantMessage, Response, Usage
from verifiers.v1.types import ToolCall as ProxyToolCall

from qorl.agent.prompts import system_prompt
from qorl.agent.tools import agent_tools
from qorl.experiment.create import latest_template
from qorl.experiment.schemas import (
    EvaluationExperimentConfig,
    ExperimentMethod,
    load_config,
)
from qorl.model import client
from qorl.model.client import HttpTransport, LocalModelClient
from qorl.model.exceptions import ContextBudgetError, ModelError, ModelRequestError
from qorl.model.schemas import (
    GenerationRequest,
    JsonObject,
    LocalDecodingSettings,
    Message,
    MessageRole,
    ModelProvider,
    ModelSettings,
    RetrySettings,
    ToolDefinition,
    ToolFunction,
)

CONTEXT_LENGTH = 20_480
OUTPUT_TOKENS = 2_048
PROMPT_TOKENS = 100
CACHED_TOKENS = 40
COMPLETION_TOKENS = 12
REASONING_TOKENS = 8
RAW_ARGUMENTS = '{ "setting": "enable_hashjoin" }'


class ScriptedTransport:
    """Return queued API replies and capture the exact payload passed to transport."""

    def __init__(self, replies: list[JsonObject]) -> None:
        self.replies = deque(replies)
        self.calls: list[tuple[str, JsonObject | None]] = []

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
        self.calls.append((path, copy.deepcopy(body)))
        return self.replies.popleft()


@pytest.fixture
def config() -> EvaluationExperimentConfig:
    result = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(result, EvaluationExperimentConfig)
    return result.model_copy(
        update={"model": result.model.model_copy(update={"name_or_path": "test-model"})}
    )


@pytest.fixture
def request_turn() -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="Use the supplied tools."),
            Message(role=MessageRole.USER, content="Inspect the setting."),
        ],
        tools=[
            ToolDefinition(
                function=ToolFunction(
                    name="inspect_setting",
                    description="Inspect one setting.",
                    parameters={
                        "type": "object",
                        "properties": {"setting": {"type": "string"}},
                    },
                )
            )
        ],
        seed=42,
    )


def completion(*, truncated: bool = False) -> JsonObject:
    """A vLLM 0.28 reply uses reasoning, not the template's reasoning_content."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Inspect first.",
                    "tool_calls": None
                    if truncated
                    else [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "inspect_setting",
                                "arguments": RAW_ARGUMENTS,
                            },
                        }
                    ],
                },
                "finish_reason": "length" if truncated else "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 8},
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }


def local_client(
    config: EvaluationExperimentConfig, transport: ScriptedTransport
) -> LocalModelClient:
    assert isinstance(config.decoding, LocalDecodingSettings)
    return LocalModelClient(config.model, config.decoding, transport=transport)


def proxy_completion() -> JsonObject:
    """Use the installed training proxy's serializer, not a hand-built response."""
    response = Response(
        id="proxy-reply",
        created=0,
        model="test-model",
        message=AssistantMessage(
            content="",
            reasoning_content="Inspect first.",
            tool_calls=[
                ProxyToolCall(
                    id="call-1", name="inspect_setting", arguments=RAW_ARGUMENTS
                )
            ],
        ),
        finish_reason="tool_calls",
        usage=Usage(
            prompt_tokens=PROMPT_TOKENS - CACHED_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            reasoning_tokens=REASONING_TOKENS,
            cached_input_tokens=CACHED_TOKENS,
        ),
    )
    return client.JSON_OBJECT.validate_python(
        proxy_train.serialize_completion(response, response.model)
    )


def test_proxy_completions_use_the_real_server_for_token_counts(
    config: EvaluationExperimentConfig,
    request_turn: GenerationRequest,
) -> None:
    assert isinstance(config.decoding, LocalDecodingSettings)
    proxy = ScriptedTransport([completion()])
    tokenizer = ScriptedTransport(
        [{"count": PROMPT_TOKENS, "max_model_len": CONTEXT_LENGTH}]
    )
    model = LocalModelClient(
        config.model,
        config.decoding,
        transport=proxy,
        token_transport=tokenizer,
        served_model_name="intercepted-model",
    )
    result = model.generate(request_turn.model_copy(update={"seed": None}))
    assert [path for path, _ in proxy.calls] == ["chat/completions"]
    assert [path for path, _ in tokenizer.calls] == ["../tokenize"]
    counted = tokenizer.calls[0][1]
    assert counted is not None
    assert counted["messages"] == result.request["messages"]
    assert counted["model"] == result.request["model"] == "intercepted-model"
    assert "seed" not in result.request


def test_real_agent_tools_and_prompt_are_not_rewritten(
    config: EvaluationExperimentConfig,
) -> None:
    definitions = agent_tools(["a", "b"])
    prompt = system_prompt(1)
    request = GenerationRequest(
        messages=[Message(role=MessageRole.SYSTEM, content=prompt)],
        tools=[ToolDefinition.model_validate(tool) for tool in definitions],
    )
    body = local_client(config, ScriptedTransport([])).request_body(request)
    assert body["tools"] == definitions
    assert body["messages"] == [{"role": "system", "content": prompt}]


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("reply", [completion, proxy_completion], ids=["vllm", "proxy"])
def test_reasoning_and_raw_arguments_survive_tool_continuation(
    config: EvaluationExperimentConfig,
    request_turn: GenerationRequest,
    thinking: bool,
    reply: Callable[[], JsonObject],
) -> None:
    assert isinstance(config.decoding, LocalDecodingSettings)
    config = config.model_copy(
        update={"decoding": config.decoding.model_copy(update={"thinking": thinking})}
    )
    count: JsonObject = {"count": PROMPT_TOKENS, "max_model_len": CONTEXT_LENGTH}
    transport = ScriptedTransport([count, reply(), count, reply()])
    model = local_client(config, transport)
    first = model.generate(request_turn)
    assert first.message.reasoning_content == "Inspect first."
    assert first.message.tool_calls is not None
    assert first.message.tool_calls[0].function.arguments == RAW_ARGUMENTS
    assert first.usage.reasoning_tokens == 8
    assert first.usage.cached_tokens == 40
    tool_result = Message(
        role=MessageRole.TOOL, tool_call_id="call-1", content='{"value": "on"}'
    )
    continued = request_turn.model_copy(
        update={"messages": [*request_turn.messages, first.message, tool_result]}
    )
    second = model.generate(continued)
    counted = transport.calls[2][1]
    assert counted is not None
    assert counted["messages"] == second.request["messages"]
    assert counted["tools"] == second.request["tools"] == first.request["tools"]
    assert counted["chat_template_kwargs"] == {"enable_thinking": thinking}
    messages = second.request["messages"]
    assert isinstance(messages, list)
    assert messages[-2] == {
        "role": "assistant",
        "content": "",
        "reasoning": "Inspect first.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "inspect_setting",
                    "arguments": RAW_ARGUMENTS,
                },
            }
        ],
    }
    assert messages[-1] == tool_result.model_dump(mode="json", exclude_none=True)
    assert second.request["max_tokens"] == OUTPUT_TOKENS
    assert second.request["seed"] == request_turn.seed
    assert second.raw_response == reply()
    assert not second.truncated


@pytest.mark.parametrize("excess", [0, 1])
def test_full_rendered_prompt_plus_output_must_fit(
    config: EvaluationExperimentConfig,
    request_turn: GenerationRequest,
    excess: int,
) -> None:
    prompt_tokens = CONTEXT_LENGTH - OUTPUT_TOKENS + excess
    transport = ScriptedTransport(
        [{"count": prompt_tokens, "max_model_len": CONTEXT_LENGTH}, completion()]
    )
    model = local_client(config, transport)
    if excess:
        with pytest.raises(ContextBudgetError) as caught:
            model.generate(request_turn)
        assert caught.value.prompt_tokens == prompt_tokens
        assert caught.value.max_tokens == OUTPUT_TOKENS
        assert [path for path, _ in transport.calls] == ["../tokenize"]
    else:
        assert model.generate(request_turn).prompt_tokens == prompt_tokens
        assert [path for path, _ in transport.calls] == [
            "../tokenize",
            "chat/completions",
        ]


def test_truncated_reasoning_is_retained_not_called_a_malformed_action(
    config: EvaluationExperimentConfig,
    request_turn: GenerationRequest,
) -> None:
    transport = ScriptedTransport(
        [
            {"count": PROMPT_TOKENS, "max_model_len": CONTEXT_LENGTH},
            completion(truncated=True),
        ]
    )
    response = local_client(config, transport).generate(request_turn)
    assert response.truncated
    assert response.finish_reason == "length"
    assert response.message.reasoning_content == "Inspect first."
    assert response.message.tool_calls is None
    assert len(transport.calls) == 2


def test_missing_usage_stays_unknown(
    config: EvaluationExperimentConfig, request_turn: GenerationRequest
) -> None:
    reply = completion()
    del reply["usage"]
    transport = ScriptedTransport(
        [{"count": PROMPT_TOKENS, "max_model_len": CONTEXT_LENGTH}, reply]
    )
    response = local_client(config, transport).generate(request_turn)
    assert response.usage.prompt_tokens is None
    assert response.usage.completion_tokens is None
    assert response.usage.reasoning_tokens is None


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"count": -1, "max_model_len": CONTEXT_LENGTH},
        {"count": 1, "max_model_len": 1},
    ],
)
def test_bad_tokenizer_reply_prevents_generation(
    config: EvaluationExperimentConfig,
    request_turn: GenerationRequest,
    reply: JsonObject,
) -> None:
    transport = ScriptedTransport([reply])
    with pytest.raises(ModelError):
        local_client(config, transport).generate(request_turn)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("adapter", [False, True])
def test_preflight_checks_parent_context_and_records_actual_version(
    config: EvaluationExperimentConfig, adapter: bool
) -> None:
    data: JsonObject = {
        "data": [{"id": "test-model", "root": "/base", "max_model_len": CONTEXT_LENGTH}]
    }
    if adapter:
        data = {
            "data": [
                {"id": "test-model", "root": "/adapter", "parent": "base"},
                {"id": "base", "root": "/base", "max_model_len": CONTEXT_LENGTH},
            ]
        }
    identity = local_client(
        config, ScriptedTransport([data, {"version": "actual-runtime-version"}])
    ).preflight()
    assert identity.vllm_version == "actual-runtime-version"
    assert identity.model.id == "test-model"
    assert identity.context_model.root == "/base"


@pytest.mark.parametrize(
    "reply",
    [
        {"data": []},
        {"data": [{"id": "test-model", "max_model_len": 1}]},
        {"data": [{"id": "test-model", "parent": "absent"}]},
        {},
    ],
)
def test_preflight_rejects_wrong_or_missing_identity(
    config: EvaluationExperimentConfig, reply: JsonObject
) -> None:
    with pytest.raises(ModelError):
        local_client(config, ScriptedTransport([reply])).preflight()


def test_local_client_rejects_hosted_model(config: EvaluationExperimentConfig) -> None:
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"provider": ModelProvider.OPENAI})
        }
    )
    with pytest.raises(ValueError, match="provider=local"):
        local_client(config, ScriptedTransport([]))


@pytest.mark.parametrize(
    "address",
    [
        "file:///tmp/model",
        "https://secret@example.com",
        "https://example.com/?key=secret",
        "http://localhost:bad",
        "http://localhost:70000",
    ],
)
def test_model_rejects_unsafe_or_malformed_addresses(
    address: str, config: EvaluationExperimentConfig
) -> None:
    with pytest.raises(ValidationError):
        ModelSettings.model_validate({**config.model.model_dump(), "base_url": address})


@pytest.mark.parametrize("field", ["base_url", "request_timeout_seconds"])
def test_clients_require_model_api_settings(
    config: EvaluationExperimentConfig, field: str
) -> None:
    assert isinstance(config.decoding, LocalDecodingSettings)
    model = config.model.model_copy(update={field: None})
    with pytest.raises(ValueError, match=field):
        HttpTransport(model)
    with pytest.raises(ValueError, match=field):
        LocalModelClient(model, config.decoding, transport=ScriptedTransport([]))


def test_transport_resolves_credentials_only_at_request_time(
    monkeypatch: pytest.MonkeyPatch,
    config: EvaluationExperimentConfig,
) -> None:
    model = config.model.model_copy(
        update={
            "base_url": "http://localhost:8000/v1",
            "request_timeout_seconds": 1,
            "api_key_env": "QORL_TEST_MODEL_KEY",
        }
    )
    transport = HttpTransport(model)
    opened = Mock(return_value=io.BytesIO(b'{"result": true}'))
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.delenv("QORL_TEST_MODEL_KEY", raising=False)
    with pytest.raises(ModelRequestError, match="QORL_TEST_MODEL_KEY"):
        transport.request("models")
    opened.assert_not_called()
    monkeypatch.setenv("QORL_TEST_MODEL_KEY", "test-secret")
    assert transport.request("../tokenize", {"messages": []}) == {"result": True}
    request = opened.call_args.args[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "http://localhost:8000/tokenize"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert "test-secret" not in model.model_dump_json()


@pytest.mark.parametrize(
    "status,fatal",
    [(400, True), (401, True), (403, True), (422, True), (429, False), (529, False)],
)
def test_http_errors_are_classified_and_credentials_redacted(
    status: int,
    fatal: bool,
    monkeypatch: pytest.MonkeyPatch,
    config: EvaluationExperimentConfig,
) -> None:
    error = urllib.error.HTTPError(
        "http://localhost",
        status,
        "failure",
        HTTPMessage(),
        io.BytesIO(b"test-secret rejected"),
    )
    opened = Mock(side_effect=error)
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    transport = HttpTransport(
        config.model.model_copy(
            update={
                "base_url": "http://localhost",
                "request_timeout_seconds": 1,
                "retry": RetrySettings(max_attempts=1),
            }
        ),
        api_key="test-secret",
    )
    with pytest.raises(ModelError) as caught:
        transport.request("models")
    assert isinstance(caught.value, ModelRequestError) is fatal
    assert "test-secret" not in str(caught.value)
    assert "[redacted]" in str(caught.value)
    assert opened.call_count == 1
