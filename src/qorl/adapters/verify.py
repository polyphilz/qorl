from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

from qorl.adapters.model import adapter_rank
from qorl.util.hashing import sha256_file
from qorl.util.serving import ServedModel

BASE_MODEL = "qorl-base"
ADAPTER_MODEL = "qorl-protocol-adapter"
MINIMUM_LOGPROBABILITY_DELTA = 1e-7
TOP_LOGPROBS = 20


def verify_merged_model(base: Path, adapter: Path, merged: Path) -> None:
    manifest_path = merged / "qorl-merge.json"
    model_path = merged / "model.safetensors"
    if not manifest_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"merged SFT model is incomplete: {merged}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "base_model_sha256": sha256_file(base / "model.safetensors"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "merged_model_sha256": sha256_file(model_path),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("merged SFT model checksum mismatch")


def request(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    call = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(call, timeout=30) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def completion(base_url: str, model: str, prompt: list[int]) -> dict[str, Any]:
    response = request(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 0,
            "logprobs": TOP_LOGPROBS,
        },
    )
    choice = response["choices"][0]
    return {
        "text": choice["text"],
        "token_logprobs": choice["logprobs"]["token_logprobs"],
        "top_logprobs": choice["logprobs"]["top_logprobs"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reload a Prime-RL adapter in vLLM and compare probabilities."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--startup-timeout", type=int, default=600)
    arguments = parser.parse_args()

    vllm = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    if not Path(vllm).is_file():
        raise RuntimeError("vLLM executable is absent from the training environment")
    audit = json.loads(arguments.audit.read_text())
    prompt = audit["probability_probe"]["prompt_token_ids"]
    target = audit["probability_probe"]["target_token"]
    if not prompt:
        raise RuntimeError("probability probe has an empty prompt")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".vllm.log")
    command = [
        vllm,
        "serve",
        str(arguments.model),
        "--served-model-name",
        BASE_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(arguments.port),
        "--language-model-only",
        "--enable-lora",
        "--max-lora-rank",
        str(adapter_rank(arguments.adapter)),
        "--lora-modules",
        f"{ADAPTER_MODEL}={arguments.adapter}",
        "--max-model-len",
        str(audit["packed_sequence_length"]),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--generation-config",
        "vllm",
        "--enforce-eager",
        "--disable-log-stats",
    ]
    environment = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    base_url = f"http://127.0.0.1:{arguments.port}"
    with ServedModel(
        command,
        repository=Path.cwd(),
        log_path=log_path,
        health_url=f"{base_url}/health",
        startup_timeout=arguments.startup_timeout,
        environment=environment,
    ):
        base = completion(base_url, BASE_MODEL, prompt)
        adapted = completion(base_url, ADAPTER_MODEL, prompt)

    keys = set(base["top_logprobs"]) | set(adapted["top_logprobs"])
    shared = keys & set(base["top_logprobs"]) & set(adapted["top_logprobs"])
    largest_shared_delta = max(
        (
            abs(base["top_logprobs"][key] - adapted["top_logprobs"][key])
            for key in shared
        ),
        default=0.0,
    )
    changed = (
        base["text"] != adapted["text"]
        or set(base["top_logprobs"]) != set(adapted["top_logprobs"])
        or largest_shared_delta > MINIMUM_LOGPROBABILITY_DELTA
    )
    if not changed:
        raise RuntimeError("reloaded adapter did not change the measured distribution")
    if target not in base["top_logprobs"] or target not in adapted["top_logprobs"]:
        raise RuntimeError(
            f"target token {target!r} is absent from top log probabilities"
        )

    report = {
        "schema_version": 1,
        "adapter_reloaded": True,
        "probabilities_changed": True,
        "prompt_tokens": len(prompt),
        "largest_shared_logprob_delta": largest_shared_delta,
        "target_token": target,
        "target_logprob_before": base["top_logprobs"][target],
        "target_logprob_after": adapted["top_logprobs"][target],
        "target_logprob_delta": (
            adapted["top_logprobs"][target] - base["top_logprobs"][target]
        ),
        "base": base,
        "adapted": adapted,
        "vllm_log": str(log_path),
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
