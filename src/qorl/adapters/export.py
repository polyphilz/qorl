from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import torch
from safetensors import torch as safetensors
from torch.distributed.checkpoint import state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from qorl.adapters.schemas import AdapterConfig, AdapterExportManifest, LoraSettings
from qorl.model.files import model_weights_sha256
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json


class TensorWriter(Protocol):
    """Safetensors' tensor-only file output boundary."""

    def save_file(
        self, tensors: dict[str, torch.Tensor], filename: Path, metadata: dict[str, str]
    ) -> None: ...


class CheckpointLoader(Protocol):
    """Load only the explicitly requested native model tensors."""

    def load(
        self,
        state_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
        *,
        checkpoint_id: Path,
    ) -> None: ...


TENSOR_WRITER: TensorWriter = safetensors
CHECKPOINT_LOADER: CheckpointLoader = state_dict_loader


def checkpoint_sha256(checkpoint: Path) -> str:
    """Bind an export to native tensor shards, excluding RNG and dataloader files."""
    files = [checkpoint / ".metadata", *sorted(checkpoint.glob("*.distcp"))]
    if len(files) == 1:
        raise ValueError("checkpoint has no tensor shards")
    return sha256_json({path.name: sha256_file(path) for path in files})


def adapter_name(checkpoint_name: str) -> str:
    name = checkpoint_name.removeprefix("app.model.")
    if name.endswith(".lora_A.0"):
        return name.removesuffix(".0") + ".weight"
    if name.endswith(".lora_B.0"):
        return name.removesuffix(".0") + ".weight"
    raise RuntimeError(f"unrecognized LoRA checkpoint key: {checkpoint_name}")


def export_configuration(
    model: Path, lora: LoraSettings, tensor_names: Iterable[str]
) -> AdapterConfig:
    """Build adapter metadata from recorded settings and the actual LoRA tensors."""
    return AdapterConfig.model_validate(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": str(model.resolve()),
            "r": lora.rank,
            "lora_alpha": lora.alpha,
            "lora_dropout": lora.dropout,
            "bias": "none",
            "target_modules": sorted(
                {
                    key.split(".")[-3]
                    for key in tensor_names
                    if key.endswith(("lora_A.weight", "lora_B.weight"))
                }
            ),
            "modules_to_save": None,
        }
    )


def export_adapter(
    checkpoint: Path, model: Path, lora: LoraSettings, output: Path
) -> AdapterExportManifest:
    """Export native LoRA tensors with the base and settings recorded by training."""
    if output.exists():
        raise ValueError(f"adapter output already exists: {output}")
    model = model.resolve()
    base_model_sha256 = model_weights_sha256(model)

    metadata = FileSystemReader(checkpoint).read_metadata()
    checkpoint_keys = sorted(
        key
        for key in metadata.state_dict_metadata
        if key.startswith("app.model.") and (".lora_A." in key or ".lora_B." in key)
    )
    if not checkpoint_keys:
        raise RuntimeError("checkpoint contains no model LoRA tensors")

    tensors: dict[str, torch.Tensor] = {}
    for key in checkpoint_keys:
        item = metadata.state_dict_metadata[key]
        if not isinstance(item, TensorStorageMetadata):
            raise RuntimeError(f"LoRA checkpoint key is not a tensor: {key}")
        tensors[key.removeprefix("app.model.")] = torch.empty(
            tuple(item.size), dtype=item.properties.dtype
        )
    state = {"app": {"model": tensors}}
    CHECKPOINT_LOADER.load(state, checkpoint_id=checkpoint)
    adapter = {
        adapter_name(f"app.model.{key}"): value.contiguous()
        for key, value in state["app"]["model"].items()
    }
    changed = sum(
        int(torch.count_nonzero(value).item())
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
    if rank != lora.rank:
        raise RuntimeError("checkpoint rank differs from recorded training rank")

    adapter_config = export_configuration(model, lora, adapter)
    native_sha256 = checkpoint_sha256(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".qorl-export-", dir=output.parent) as temporary:
        staging = Path(temporary)
        weights = staging / "adapter_model.safetensors"
        TENSOR_WRITER.save_file(adapter, weights, metadata={"format": "pt"})
        write_json(
            staging / "adapter_config.json", adapter_config.model_dump(mode="json")
        )
        manifest = AdapterExportManifest(
            tensor_count=len(adapter),
            nonzero_lora_b_values=changed,
            adapter_sha256=sha256_file(weights),
            base_model_sha256=base_model_sha256,
            checkpoint_sha256=native_sha256,
        )
        write_json(staging / "qorl-manifest.json", manifest.model_dump(mode="json"))
        staging.rename(output)
    return manifest


def main() -> None:
    """Export using a saved native SFT configuration, not independent scale flags."""
    from prime_rl.configs.sft import SFTConfig

    parser = argparse.ArgumentParser(
        description="Export a recorded SFT checkpoint as a PEFT adapter."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = SFTConfig.model_validate_json(Path(arguments.training_config).read_bytes())
    if config.model.lora is None:
        raise ValueError("recorded training configuration has no LoRA settings")
    settings = config.model.lora
    manifest = export_adapter(
        Path(arguments.checkpoint),
        Path(config.model.name),
        LoraSettings(
            rank=settings.rank,
            alpha=settings.alpha,
            dropout=settings.dropout,
            target_modules=settings.target_modules,
        ),
        Path(arguments.output),
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
