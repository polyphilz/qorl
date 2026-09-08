"""Serve a selected complete model, optionally with its verified LoRA adapter."""

import json
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from qorl.adapters.config import adapter_rank
from qorl.adapters.verify import verify_adapter_base
from qorl.inference.serving import ServedModel
from qorl.model.client import LocalModelClient
from qorl.model.exceptions import ModelError
from qorl.model.files import resolve_model
from qorl.model.schemas import LocalInferenceSettings, ModelSettings
from qorl.paths import REPOSITORY_ROOT

BASE_MODEL_NAME = "qorl-base"
ADAPTER_MODEL_NAME = "qorl-adapter"


def server_command(
    model: ModelSettings,
    inference: LocalInferenceSettings,
    gpu_ids: list[int],
) -> list[str]:
    """Resolve weights and validate adapter provenance before launching vLLM."""
    serving = inference.serving
    base = resolve_model(model)
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError("serving requires distinct, nonnegative GPU IDs")
    if inference.max_tokens > model.context_length:
        raise ValueError("inference.max_tokens exceeds model.context_length")
    if inference.thinking and serving.reasoning_parser is None:
        raise ValueError("thinking requires inference.serving.reasoning_parser")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(base),
        "--served-model-name",
        BASE_MODEL_NAME,
        "--host",
        serving.host,
        "--port",
        str(serving.port),
        "--dtype",
        serving.dtype,
        "--max-model-len",
        str(model.context_length),
        "--max-num-seqs",
        str(serving.max_num_seqs),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(len(gpu_ids)),
        "--tool-call-parser",
        serving.tool_call_parser,
        "--enable-auto-tool-choice",
        "--generation-config",
        "vllm",
        "--language-model-only",
        "--enforce-eager",
        "--enable-prefix-caching"
        if serving.enable_prefix_caching
        else "--no-enable-prefix-caching",
    ]
    if serving.reasoning_parser is not None:
        command.extend(["--reasoning-parser", serving.reasoning_parser])
    if model.adapter_path is not None:
        adapter = (REPOSITORY_ROOT / model.adapter_path.expanduser()).resolve()
        verify_adapter_base(adapter, base)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError(f"adapter weights missing: {adapter}")
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(adapter_rank(adapter)),
                "--lora-modules",
                json.dumps(
                    {
                        "name": ADAPTER_MODEL_NAME,
                        "path": str(adapter),
                        "base_model_name": BASE_MODEL_NAME,
                    }
                ),
            ]
        )
    return command


@contextmanager
def serve_local_model(
    model: ModelSettings,
    inference: LocalInferenceSettings,
    gpu_ids: list[int],
    log_path: Path,
) -> Generator[LocalModelClient]:
    """Own server startup/cleanup and yield a preflight-checked connection."""
    serving = inference.serving
    if model.base_url is None or model.request_timeout_seconds is None:
        raise ValueError(
            "local serving requires model.base_url and model.request_timeout_seconds"
        )
    address = urlsplit(model.base_url)
    host = "127.0.0.1" if serving.host == "0.0.0.0" else serving.host
    if (
        address.scheme != "http"
        or address.hostname != host
        or address.port != serving.port
        or address.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "model.base_url must address the configured local server's /v1 endpoint"
        )
    if model.api_key_env is not None:
        raise ValueError("managed local serving does not use an API credential")
    command = server_command(model, inference, gpu_ids)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
        "VLLM_ENFORCE_STRICT_TOOL_CALLING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "1" if serving.use_flashinfer_sampler else "0",
    }
    client = LocalModelClient(
        model,
        inference,
        served_model_name=ADAPTER_MODEL_NAME if model.adapter_path else BASE_MODEL_NAME,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ServedModel(
        command,
        repository=REPOSITORY_ROOT,
        log_path=log_path,
        health_url=f"{address.scheme}://{address.netloc}/health",
        startup_timeout=serving.startup_timeout_seconds,
        environment=environment,
    ):
        identity = client.preflight()
        if identity.context_model.root != str(resolve_model(model)):
            raise ModelError("served base model path differs from the requested model")
        if model.adapter_path is not None:
            adapter = (REPOSITORY_ROOT / model.adapter_path.expanduser()).resolve()
            if identity.model.root != str(adapter):
                raise ModelError(
                    "served adapter path differs from the requested adapter"
                )
        yield client
