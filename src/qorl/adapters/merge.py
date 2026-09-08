"""Apply exported plain LoRA updates to a complete local safetensors model."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import torch
from pydantic import TypeAdapter
from safetensors import torch as safetensors

from qorl.adapters.schemas import MergeLoraConfig, MergeManifest
from qorl.adapters.verify import (
    ADAPTER_MANIFEST_FILE,
    artifact_inventory,
    tensor_signatures,
    verify_adapter_base,
    verify_adapter_weights,
    verify_merged_model,
)
from qorl.model.files import (
    model_weight_files,
    model_weights_sha256,
    validate_model_directory,
)
from qorl.model.schemas import JsonObject
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json

MANIFEST_FILE = "qorl-merge.json"
ADAPTER_FILE = "adapter_model.safetensors"
MATRIX_DIMENSIONS = 2
OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class TensorFiles(Protocol):
    """Typed tensor-only boundary for safetensors' PathLike signatures."""

    def load_file(self, filename: Path) -> dict[str, torch.Tensor]: ...

    def save_file(
        self, tensors: dict[str, torch.Tensor], filename: Path, metadata: dict[str, str]
    ) -> None: ...


TENSOR_FILES: TensorFiles = safetensors


def adapter_pairs(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for key in sorted(tensors):
        marker = ".lora_A.weight"
        if not key.endswith(marker):
            continue
        base_key = key.removesuffix(marker) + ".weight"
        b_key = key.removesuffix(marker) + ".lora_B.weight"
        if b_key not in tensors:
            raise RuntimeError(f"missing LoRA B tensor for {key}")
        pairs[base_key] = (key, b_key)
    if not pairs or len(pairs) * 2 != len(tensors):
        raise RuntimeError("adapter must contain only nonempty LoRA A/B pairs")
    return pairs


def merged_weight(
    key: str,
    weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    config: MergeLoraConfig,
) -> torch.Tensor:
    """Accumulate in float32, then retain the base's floating-point dtype."""
    if (
        weight.ndim != MATRIX_DIMENSIONS
        or a.ndim != MATRIX_DIMENSIONS
        or b.ndim != MATRIX_DIMENSIONS
        or a.shape != (config.r, weight.shape[1])
        or b.shape != (weight.shape[0], config.r)
    ):
        raise RuntimeError(f"LoRA shape/rank mismatch for {key}")
    if not all(
        t.is_floating_point() and bool(torch.isfinite(t).all()) for t in (weight, a, b)
    ):
        raise RuntimeError(f"LoRA merge requires finite floating-point tensors: {key}")
    if not any(
        key.removesuffix(".weight").endswith("." + target) or key == target + ".weight"
        for target in config.target_modules
    ):
        raise RuntimeError(f"LoRA tensor is outside target_modules: {key}")
    merged = (
        weight.float() + (config.lora_alpha / config.r) * (b.float() @ a.float())
    ).to(weight.dtype)
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError(f"LoRA update overflows base dtype: {key}")
    return merged.contiguous()


def merge(base: Path, adapter: Path, output: Path) -> Path:
    """Stage and verify a complete model, preserving the base shard boundaries."""
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output already exists: {output}")
    if any(
        output.resolve().is_relative_to(source.resolve()) for source in (base, adapter)
    ):
        raise RuntimeError(
            "merge output must be outside the base and adapter directories"
        )
    validate_model_directory(base)
    config = MergeLoraConfig.model_validate_json(
        (adapter / "adapter_config.json").read_bytes()
    )
    base_config = OBJECT.validate_json((base / "config.json").read_bytes())
    if base_config.get("quantization_config") is not None:
        raise RuntimeError("quantized bases are not supported for LoRA merging")
    base_hash = verify_adapter_base(adapter, base)
    export = verify_adapter_weights(adapter)
    files = model_weight_files(base)
    index_path = next(
        (file for file in files if file.name.endswith(".index.json")), None
    )
    shards = [file for file in files if file != index_path]
    if any(file.suffix != ".safetensors" for file in shards):
        raise RuntimeError("merge requires a safetensors base")
    base_tensors = tensor_signatures(base)
    adapter_tensors = TENSOR_FILES.load_file(adapter / ADAPTER_FILE)
    if len(adapter_tensors) != export.tensor_count:
        raise RuntimeError("adapter tensor count differs from export manifest")
    pairs = adapter_pairs(adapter_tensors)
    if missing := set(pairs) - base_tensors.keys():
        raise RuntimeError(
            f"LoRA tensors do not match base model: {sorted(missing)[:3]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for source in base.rglob("*"):
            if (
                not source.is_file()
                or source.suffix in {".safetensors", ".bin"}
                or source.name.endswith(".index.json")
            ):
                continue
            if source.name in {
                MANIFEST_FILE,
                ADAPTER_MANIFEST_FILE,
                "adapter_config.json",
            }:
                continue
            target = staging / source.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for shard in shards:
            tensors = TENSOR_FILES.load_file(shard)
            relative = shard.relative_to(base).as_posix()
            for key, weight in tensors.items():
                if key in pairs:
                    a_key, b_key = pairs[key]
                    tensors[key] = merged_weight(
                        key,
                        weight,
                        adapter_tensors[a_key],
                        adapter_tensors[b_key],
                        config,
                    )
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            TENSOR_FILES.save_file(tensors, destination, metadata={"format": "pt"})
        if index_path is not None:
            shutil.copy2(index_path, staging / index_path.name)
        manifest = MergeManifest(
            base_model_sha256=base_hash,
            adapter_model_sha256=export.adapter_sha256,
            adapter_config_sha256=sha256_file(adapter / "adapter_config.json"),
            adapter_manifest_sha256=sha256_file(adapter / ADAPTER_MANIFEST_FILE),
            merged_model_sha256=model_weights_sha256(staging),
            lora_rank=config.r,
            lora_alpha=config.lora_alpha,
            lora_scale=config.lora_alpha / config.r,
            merged_tensor_count=len(pairs),
            artifacts=artifact_inventory(staging),
        )
        write_json(staging / MANIFEST_FILE, manifest.model_dump(mode="json"))
        verify_merged_model(base, adapter, staging)
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"output already exists: {output}")
        staging.rename(output)
    return output
