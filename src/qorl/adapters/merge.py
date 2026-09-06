#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qorl.adapters.model import adapter_config, verify_adapter_base

MODEL_FILE = "model.safetensors"
ADAPTER_FILE = "adapter_model.safetensors"
MANIFEST_FILE = "qorl-merge.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_pairs(path: Path) -> dict[str, tuple[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
    pairs: dict[str, tuple[str, str]] = {}
    for key in sorted(keys):
        marker = ".lora_A.weight"
        if not key.endswith(marker):
            continue
        base_key = key[: -len(marker)] + ".weight"
        b_key = key[: -len(marker)] + ".lora_B.weight"
        if b_key not in keys:
            raise RuntimeError(f"missing LoRA B tensor for {key}")
        pairs[base_key] = (key, b_key)
    if len(pairs) * 2 != len(keys):
        raise RuntimeError("adapter contains tensors other than LoRA A/B pairs")
    return pairs


def merge(base: Path, adapter: Path, output: Path, repository: Path) -> Path:
    base_model = base / MODEL_FILE
    adapter_model = adapter / ADAPTER_FILE
    adapter_config_path = adapter / "adapter_config.json"
    for path in (base_model, adapter_model, adapter_config_path):
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    config = adapter_config(adapter)
    if config.peft_type != "LORA" or config.bias != "none":
        raise RuntimeError("only an unbiased LoRA adapter can be merged")
    base_model_sha256 = verify_adapter_base(adapter, base, repository)
    rank = config.r
    scale = config.lora_alpha / rank
    pairs = adapter_pairs(adapter_model)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        target = Path(temporary)
        for source in base.iterdir():
            if source.name not in {MODEL_FILE, MANIFEST_FILE} and source.is_file():
                shutil.copy2(source, target / source.name, follow_symlinks=True)

        with (
            safe_open(base_model, framework="pt", device="cpu") as base_file,
            safe_open(adapter_model, framework="pt", device="cpu") as lora_file,
        ):
            base_keys = set(base_file.keys())
            missing = sorted(set(pairs) - base_keys)
            if missing:
                raise RuntimeError(
                    f"LoRA tensors do not match base model: {missing[:3]}"
                )

            tensors: dict[str, torch.Tensor] = {}
            for index, key in enumerate(base_file.keys(), start=1):
                weight = base_file.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = lora_file.get_tensor(a_key).float()
                    b = lora_file.get_tensor(b_key).float()
                    if b.shape[0] != weight.shape[0] or a.shape[1] != weight.shape[1]:
                        raise RuntimeError(f"LoRA shape mismatch for {key}")
                    weight = (weight.float() + scale * (b @ a)).to(weight.dtype)
                tensors[key] = weight.contiguous()
                if index % 100 == 0:
                    print(f"loaded {index}/{len(base_keys)} base tensors", flush=True)
            save_file(
                tensors,
                target / MODEL_FILE,
                metadata=base_file.metadata(),
            )

        merged_model_sha256 = sha256(target / MODEL_FILE)
        manifest = {
            "schema_version": 1,
            "operation": "merge_lora_into_base",
            "base_model_sha256": base_model_sha256,
            "adapter_model_sha256": sha256(adapter_model),
            "adapter_config_sha256": sha256(adapter_config_path),
            "merged_model_sha256": merged_model_sha256,
            "lora_rank": rank,
            "lora_alpha": config.lora_alpha,
            "lora_scale": scale,
            "merged_tensor_count": len(pairs),
            "artifacts": [
                {
                    "path": path.relative_to(target).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": (
                        merged_model_sha256 if path.name == MODEL_FILE else sha256(path)
                    ),
                }
                for path in sorted(target.rglob("*"))
                if path.is_file()
            ],
        }
        (target / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).rename(output)
    return output / MANIFEST_FILE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        merge(
            arguments.base.resolve(),
            arguments.adapter.resolve(),
            arguments.output.resolve(),
            arguments.repository.resolve(),
        )
    )


if __name__ == "__main__":
    main()
