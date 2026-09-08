"""Real local/native parser parity using the configured snapshot and renderer."""

import asyncio
import os
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from renderers.base import ToolSpec, load_tokenizer
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from qorl.experiment.schemas import RlExperimentConfig, load_config
from qorl.model.client import JSON_OBJECT, LocalModelClient
from qorl.model.exceptions import ContextBudgetError
from qorl.model.schemas import (
    ChatResponse,
    GenerationRequest,
    JsonObject,
    LocalInferenceSettings,
    Message,
    MessageRole,
    ToolDefinition,
    ToolFunction,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_RL_GPU") != "1",
    reason="requires the agreed isolated GPU environment and model snapshot",
)


@pytest.mark.parametrize(
    "names",
    [[], ["finish"], ["finish", "finish"], ["unavailable"], ["finish", "unavailable"]],
)
def test_local_chat_and_native_preserve_tool_attempts(
    names: list[str], monkeypatch: pytest.MonkeyPatch, repository_root: Path
) -> None:
    protocol = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
    tool_calls = pytest.importorskip("vllm.entrypoints.serve.utils.tool_calls_utils")
    qwen_parser = pytest.importorskip("vllm.tool_parsers.qwen3_engine_tool_parser")
    native_train = pytest.importorskip("verifiers.v1.clients.train")

    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
    tokenizer = load_tokenizer(str(Path(os.environ["QORL_TEST_MODEL_PATH"])))
    functions: list[ToolSpec] = [
        {
            "name": "finish",
            "description": "Finish",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    tools = [{"type": "function", "function": function} for function in functions]
    loaded = load_config(repository_root / "configs/defaults/000-rl.toml")
    assert isinstance(loaded.inference, LocalInferenceSettings)
    local = LocalModelClient(
        loaded.model, loaded.inference, served_model_name="qorl-base"
    )
    request = protocol.ChatCompletionRequest.model_validate(
        local.request_body(
            GenerationRequest(
                messages=[Message(role=MessageRole.USER, content="Act")],
                tools=[ToolDefinition.model_validate(tool) for tool in tools],
            )
        )
    )
    text = (
        "\n".join(
            f"<tool_call>\n<function={name}>\n</function>\n</tool_call>"
            for name in names
        )
        or "No tool."
    )
    parser = qwen_parser.Qwen3EngineToolParser(tokenizer, request.tools)
    adjusted = parser.adjust_request(request)
    assert adjusted.structured_outputs is None
    parsed = parser.extract_tool_calls(text, adjusted)
    choice = tool_calls.maybe_filter_parallel_tool_calls(
        protocol.ChatCompletionResponseChoice(
            index=0,
            message=protocol.ChatMessage(
                role="assistant", content=parsed.content, tool_calls=parsed.tool_calls
            ),
            finish_reason="tool_calls" if parsed.tools_called else "stop",
        ),
        adjusted,
    )
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    native_parsed = renderer.parse_response(token_ids, tools=functions)
    native = native_train.response_from_generate(
        {"tool_calls": native_parsed.tool_calls, "completion_ids": token_ids},
        "qorl-base",
    )
    local_response = ChatResponse.model_validate({"choices": [choice.model_dump()]})
    native_response = ChatResponse.model_validate(
        native_train.serialize_completion(native, "qorl-base")
    )
    local_calls = local_response.choices[0].message.tool_calls or []
    native_calls = native_response.choices[0].message.tool_calls or []
    assert [call.function.name for call in local_calls] == names
    assert [call.function.name for call in native_calls] == names


@pytest.mark.parametrize("excess", [0, 1])
def test_actual_native_prompt_budget_prevents_generation(
    excess: int, monkeypatch: pytest.MonkeyPatch, repository_root: Path
) -> None:
    native = pytest.importorskip("verifiers.v1.clients.train")
    vf = pytest.importorskip("verifiers.v1")
    graph = pytest.importorskip("verifiers.v1.graph")
    configs = pytest.importorskip("verifiers.v1.configs.client")
    dialects = pytest.importorskip("verifiers.v1.dialects")
    base = os.environ["QORL_TEST_MODEL_PATH"]
    loaded = load_config(repository_root / "configs/defaults/000-rl.toml")
    assert isinstance(loaded, RlExperimentConfig)
    request = GenerationRequest(
        messages=[Message(role=MessageRole.USER, content="Finish now")],
        tools=[
            ToolDefinition(
                function=ToolFunction(
                    name="finish",
                    description="Finish",
                    parameters={"type": "object", "properties": {}},
                )
            )
        ],
    )
    tokenizer = load_tokenizer(base)
    renderer_module = pytest.importorskip("renderers.qwen35")
    renderer = renderer_module.Qwen35Renderer(
        tokenizer, Qwen35RendererConfig(enable_thinking=False)
    )
    tools = [tool.model_dump(mode="json") for tool in request.tools]
    expected = renderer.render(
        [{"role": "user", "content": "Finish now"}],
        tools=tools,
        add_generation_prompt=True,
    ).token_ids
    limit = len(expected) + 128 - excess
    generated: list[JsonObject] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "qorl-base", "max_model_len": limit}]}
            )
        assert http_request.url.path == "/inference/v1/generate"
        generated.append(JSON_OBJECT.validate_json(http_request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [],
                        "logprobs": {"content": []},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    api = AsyncOpenAI(
        base_url=f"http://budget-{excess}.test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    def build_api(_: object) -> AsyncOpenAI:
        return api

    monkeypatch.setattr(native, "build_async_openai", build_api)
    training_client = native.TrainClient(
        configs.TrainClientConfig(
            renderer_model_name=base,
            renderer={"name": "qwen3.5", "enable_thinking": False},
        )
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    sampling = vf.SamplingConfig(
        max_tokens=128, chat_template_kwargs={"enable_thinking": False}
    )
    dialect = dialects.ChatDialect()

    class Transport:
        def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
            assert body is not None
            turn = graph.prepare_turn(trace, dialect.parse_request(body).messages)
            if path == "../tokenize":
                counted = asyncio.run(
                    training_client.relay_aux(
                        dialect, "/tokenize", body, sampling=sampling, turn=turn
                    )
                )
                assert counted["count"] == len(expected)
                return JSON_OBJECT.validate_python(counted)
            response = asyncio.run(
                training_client.get_response(dialect, body, sampling, turn=turn)
            )
            return JSON_OBJECT.validate_python(response.raw)

    model = loaded.model.model_copy(update={"context_length": limit})
    inference = loaded.inference.model_copy(
        update={"max_tokens": 128, "thinking": False}
    )
    assert isinstance(inference, LocalInferenceSettings)
    client = LocalModelClient(
        model, inference, transport=Transport(), served_model_name="qorl-base"
    )
    try:
        if excess:
            with pytest.raises(ContextBudgetError):
                client.generate(request)
            assert generated == []
        else:
            assert client.generate(request).prompt_tokens == len(expected)
            assert len(generated) == 1
            assert generated[0]["token_ids"] == expected
    finally:
        asyncio.run(training_client.close())
