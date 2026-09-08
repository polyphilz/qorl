"""Server command provenance and lifecycle checks without loading GPU libraries."""

import json
import os
import socket
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from qorl.adapters.schemas import AdapterConfig
from qorl.agent.tools import agent_tools
from qorl.agent.types import ToolName
from qorl.experiment.create import latest_template
from qorl.experiment.schemas import (
    EvaluationExperimentConfig,
    ExperimentMethod,
    load_config,
)
from qorl.inference import local
from qorl.inference.local import (
    ADAPTER_MODEL_NAME,
    BASE_MODEL_NAME,
    serve_local_model,
    server_command,
)
from qorl.model.exceptions import ModelError
from qorl.model.schemas import (
    AdvertisedModel,
    GenerationRequest,
    LocalInferenceSettings,
    LocalServerIdentity,
    Message,
    MessageRole,
    ModelProvider,
    ModelWeightIndex,
)

ADAPTER_RANK = 16
LIVE_REASONING_OUTPUT_TOKENS = 16_384


@pytest.fixture
def config(tmp_path: Path) -> EvaluationExperimentConfig:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"base weights")
    result = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(result, EvaluationExperimentConfig)
    return result.model_copy(
        update={
            "model": result.model.model_copy(
                update={"name_or_path": str(base), "revision": None}
            )
        }
    )


@pytest.fixture
def adapter(config: EvaluationExperimentConfig, tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        AdapterConfig(
            base_model_name_or_path=config.model.name_or_path,
            peft_type="LORA",
            bias="none",
            r=ADAPTER_RANK,
            lora_alpha=32,
        ).model_dump_json()
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter weights")
    return path


@pytest.mark.parametrize("thinking", [False, True])
def test_command_uses_selected_weights_context_gpu_count_and_parsers(
    config: EvaluationExperimentConfig, thinking: bool
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    inference = config.inference.model_copy(update={"thinking": thinking})
    command = server_command(config.model, inference, [0, 1])
    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == config.model.name_or_path
    assert command[command.index("--max-model-len") + 1] == str(
        config.model.context_length
    )
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-lora" not in command


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("sharded", [False, True])
def test_adapter_is_applied_to_its_verified_training_base(
    config: EvaluationExperimentConfig,
    adapter: Path,
    filename: str,
    sharded: bool,
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    model = config.model.model_copy(update={"adapter_path": adapter})
    base = Path(model.name_or_path)
    (base / "model.safetensors").unlink()
    if sharded:
        (base / f"first-{filename}").write_bytes(b"first shard")
        (base / f"second-{filename}").write_bytes(b"second shard")
        (base / f"{filename}.index.json").write_text(
            ModelWeightIndex(
                weight_map={
                    "a": f"first-{filename}",
                    "b": f"second-{filename}",
                }
            ).model_dump_json()
        )
    else:
        (base / filename).write_bytes(b"base weights")
    command = server_command(model, config.inference, [0])
    assert command[command.index("--max-lora-rank") + 1] == str(ADAPTER_RANK)
    assert json.loads(command[command.index("--lora-modules") + 1]) == {
        "name": ADAPTER_MODEL_NAME,
        "path": str(adapter),
        "base_model_name": BASE_MODEL_NAME,
    }


def test_unconfigured_parser_is_omitted_when_thinking_is_off(
    config: EvaluationExperimentConfig,
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    command = server_command(
        config.model,
        config.inference.model_copy(
            update={
                "thinking": False,
                "serving": config.inference.serving.model_copy(
                    update={"reasoning_parser": None}
                ),
            }
        ),
        [0],
    )
    assert "--reasoning-parser" not in command


def test_wrong_base_is_rejected_before_any_server_starts(
    config: EvaluationExperimentConfig, adapter: Path, tmp_path: Path
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    other = tmp_path / "wrong-base"
    other.mkdir()
    (other / "config.json").write_text("{}")
    (other / "model.safetensors").write_bytes(b"different weights")
    model = config.model.model_copy(
        update={"adapter_path": adapter, "name_or_path": str(other)}
    )
    with pytest.raises(
        RuntimeError, match="does not match the adapter's training base"
    ):
        server_command(model, config.inference, [0])


def test_command_resolves_only_the_pinned_hf_cache_revision(
    config: EvaluationExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    cache = tmp_path / "cache"
    snapshot = cache / "models--organization--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"base weights")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    model = config.model.model_copy(
        update={"name_or_path": "organization/model", "revision": "pinned-revision"}
    )
    command = server_command(model, config.inference, [0])
    assert command[command.index("--model") + 1] == str(snapshot)
    with pytest.raises(RuntimeError, match="pinned model snapshot is missing"):
        server_command(
            model.model_copy(update={"revision": "missing"}),
            config.inference,
            [0],
        )


@pytest.mark.parametrize("gpu_ids", [[], [0, 0], [-1]])
def test_invalid_gpu_allocation_is_rejected(
    config: EvaluationExperimentConfig, gpu_ids: list[int]
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    with pytest.raises(ValueError, match="GPU IDs"):
        server_command(config.model, config.inference, gpu_ids)


def test_thinking_requires_an_explicit_parser(
    config: EvaluationExperimentConfig,
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    with pytest.raises(ValueError, match="reasoning_parser"):
        server_command(
            config.model,
            config.inference.model_copy(
                update={
                    "thinking": True,
                    "serving": config.inference.serving.model_copy(
                        update={"reasoning_parser": None}
                    ),
                }
            ),
            [0],
        )


def test_hosted_model_cannot_launch_a_local_server(
    config: EvaluationExperimentConfig,
) -> None:
    assert isinstance(config.inference, LocalInferenceSettings)
    with pytest.raises(ValueError, match="hosted models"):
        server_command(
            config.model.model_copy(update={"provider": ModelProvider.OPENAI}),
            config.inference,
            [0],
        )


@pytest.mark.parametrize("failure", [None, "preflight", "body", "wrong-root"])
@pytest.mark.parametrize("with_adapter", [False, True])
def test_server_scope_cleans_up_and_checks_real_advertised_paths(
    config: EvaluationExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    with_adapter: bool,
    adapter: Path,
) -> None:
    if with_adapter:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"adapter_path": adapter})}
        )
    assert (
        isinstance(config.inference, LocalInferenceSettings)
        and config.model.base_url is not None
    )
    process = MagicMock()
    process.__exit__.return_value = False
    factory = Mock(return_value=process)
    monkeypatch.setattr(local, "ServedModel", factory)
    root = "/wrong-base" if failure == "wrong-root" else config.model.name_or_path
    base = AdvertisedModel(
        id=BASE_MODEL_NAME, root=root, max_model_len=config.model.context_length
    )
    selected = (
        AdvertisedModel(
            id=ADAPTER_MODEL_NAME, root=str(adapter), parent=BASE_MODEL_NAME
        )
        if with_adapter
        else base
    )
    identity = LocalServerIdentity(
        base_url=config.model.base_url,
        model=selected,
        context_model=base,
        vllm_version="0.28.0",
    )
    preflight = Mock(
        return_value=identity,
        side_effect=ModelError("preflight failed") if failure == "preflight" else None,
    )
    monkeypatch.setattr(local.LocalModelClient, "preflight", preflight)
    with (
        pytest.raises((ModelError, RuntimeError)) if failure else nullcontext(),
        serve_local_model(
            config.model,
            config.inference,
            [1],
            tmp_path / "logs/server.log",
        ) as model,
    ):
        assert model.served_model_name == selected.id
        if failure == "body":
            raise RuntimeError("evaluation failed")
    process.__enter__.assert_called_once()
    process.__exit__.assert_called_once()
    environment = factory.call_args.kwargs["environment"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert factory.call_args.kwargs["health_url"] == "http://127.0.0.1:8000/health"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8001/v1",
        "https://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/other",
    ],
)
def test_connection_cannot_point_to_another_server(
    config: EvaluationExperimentConfig, tmp_path: Path, endpoint: str
) -> None:
    assert (
        isinstance(config.inference, LocalInferenceSettings)
        and config.model.base_url is not None
    )
    with (
        pytest.raises(ValueError, match=r"model\.base_url"),
        serve_local_model(
            config.model.model_copy(update={"base_url": endpoint}),
            config.inference,
            [0],
            tmp_path / "server.log",
        ),
    ):
        pytest.fail("invalid connection must not start a server")


@pytest.mark.parametrize("with_adapter", [False, True], ids=["base", "adapter"])
@pytest.mark.parametrize("thinking", [False, True], ids=["thinking-off", "thinking-on"])
def test_live_thinking_modes_complete_and_preserve_terminal_tools(
    tmp_path: Path,
    with_adapter: bool,
    thinking: bool,
    record_property: Callable[[str, str], None],
) -> None:
    """Opt in with QORL_TEST_MODEL_PATH and, for the adapter case, QORL_TEST_ADAPTER_PATH."""
    model_path = os.environ.get("QORL_TEST_MODEL_PATH")
    adapter_path = os.environ.get("QORL_TEST_ADAPTER_PATH") if with_adapter else None
    if model_path is None or (with_adapter and adapter_path is None):
        pytest.skip(
            "live local-model test requires QORL_TEST_MODEL_PATH and the selected adapter path"
        )
    config = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(config, EvaluationExperimentConfig)
    assert isinstance(config.inference, LocalInferenceSettings)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    model = config.model.model_copy(
        update={
            "name_or_path": model_path,
            "revision": None,
            "adapter_path": Path(adapter_path) if adapter_path else None,
            "base_url": f"http://127.0.0.1:{port}/v1",
        }
    )
    inference = config.inference.model_copy(
        update={
            "thinking": thinking,
            "temperature": 0,
            "max_tokens": LIVE_REASONING_OUTPUT_TOKENS
            if thinking
            else config.inference.max_tokens,
            "serving": config.inference.serving.model_copy(
                update={"host": "127.0.0.1", "port": port}
            ),
        }
    )
    gpu_ids = [int(os.environ.get("QORL_TEST_GPU_ID", "0"))]
    with serve_local_model(
        model, inference, gpu_ids, tmp_path / "server.log"
    ) as client:
        assert client.identity is not None
        record_property("server_identity", client.identity.model_dump_json())
        history: list[Message] = []
        for name in (ToolName.KEEP_DEFAULT, ToolName.FINISH):
            tool = next(
                item for item in agent_tools(["a"]) if item.function.name == name.value
            )
            response = client.generate(
                GenerationRequest(
                    messages=[
                        *history,
                        Message(
                            role=MessageRole.USER,
                            content=f"Call {name.value} now with no arguments.",
                        ),
                    ],
                    tools=[tool],
                    seed=0,
                )
            )
            evidence = response.model_dump_json(indent=2)
            (tmp_path / f"{name.value}.json").write_text(evidence)
            record_property(name.value, evidence)
            assert response.request["chat_template_kwargs"] == {
                "enable_thinking": thinking
            }
            assert not response.truncated
            assert response.finish_reason == "tool_calls"
            if not thinking:
                assert response.usage.reasoning_tokens == 0
                assert not (response.message.reasoning_content or "").strip()
            assert response.usage.completion_tokens is not None
            assert response.usage.completion_tokens < response.requested_max_tokens
            assert response.message.tool_calls is not None
            assert len(response.message.tool_calls) == 1
            call = response.message.tool_calls[0]
            assert call.function.name == name.value
            assert json.loads(call.function.arguments) == {}
            if history and history[1].reasoning_content is not None:
                messages = response.request["messages"]
                assert isinstance(messages, list)
                previous = messages[1]
                assert isinstance(previous, dict)
                assert previous["reasoning"] == history[1].reasoning_content
            history.extend(
                [
                    Message(
                        role=MessageRole.USER,
                        content=f"Call {name.value} now with no arguments.",
                    ),
                    response.message,
                    Message(
                        role=MessageRole.TOOL,
                        tool_call_id=call.id,
                        name=name.value,
                        content='{"status":"completed"}',
                    ),
                ]
            )
