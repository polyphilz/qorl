from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load

from qorl.adapters.schemas import AdapterExportManifest
from qorl.util.hashing import sha256_file

MODEL_FILE = "model.safetensors"


def adapter_name(checkpoint_name: str) -> str:
    name = checkpoint_name.removeprefix("app.model.")
    if name.endswith(".lora_A.0"):
        return name.removesuffix(".0") + ".weight"
    if name.endswith(".lora_B.0"):
        return name.removesuffix(".0") + ".weight"
    raise RuntimeError(f"unrecognized LoRA checkpoint key: {checkpoint_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Prime-RL LoRA checkpoint in PEFT/vLLM format."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    arguments = parser.parse_args()
    model = arguments.model.resolve()
    model_weights = model / MODEL_FILE
    if not model_weights.is_file():
        raise RuntimeError(f"base model weights are missing: {model_weights}")

    metadata = FileSystemReader(arguments.checkpoint).read_metadata()
    checkpoint_keys = sorted(
        key
        for key in metadata.state_dict_metadata
        if key.startswith("app.model.") and (".lora_A." in key or ".lora_B." in key)
    )
    if not checkpoint_keys:
        raise RuntimeError("checkpoint contains no model LoRA tensors")

    tensors = {}
    for key in checkpoint_keys:
        item = metadata.state_dict_metadata[key]
        tensors[key.removeprefix("app.model.")] = torch.empty(
            tuple(item.size), dtype=item.properties.dtype
        )
    state = {"app": {"model": tensors}}
    load(state, checkpoint_id=arguments.checkpoint)
    adapter = {
        adapter_name(f"app.model.{key}"): value.contiguous()
        for key, value in state["app"]["model"].items()
    }
    changed = sum(
        torch.count_nonzero(value).item()
        for key, value in adapter.items()
        if ".lora_B." in key
    )
    if changed == 0:
        raise RuntimeError("all LoRA B tensors remain zero after training")
    ranks = {
        value.shape[0]
        for key, value in adapter.items()
        if key.endswith("lora_A.weight")
    }
    if len(ranks) != 1:
        raise RuntimeError(f"checkpoint contains inconsistent LoRA ranks: {ranks}")
    rank = ranks.pop()

    arguments.output.mkdir(parents=True, exist_ok=True)
    weights = arguments.output / "adapter_model.safetensors"
    save_file(adapter, weights, metadata={"format": "pt"})
    target_modules = sorted(
        {
            key.split(".")[-3]
            for key in adapter
            if key.endswith(("lora_A.weight", "lora_B.weight"))
        }
    )
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": str(model),
        "r": rank,
        "lora_alpha": arguments.alpha,
        "lora_dropout": arguments.dropout,
        "bias": "none",
        "target_modules": target_modules,
        "modules_to_save": None,
    }
    (arguments.output / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    )
    manifest = AdapterExportManifest(
        tensor_count=len(adapter),
        nonzero_lora_b_values=changed,
        adapter_sha256=sha256_file(weights),
        base_model_sha256=sha256_file(model_weights),
    )
    (arguments.output / "qorl-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
