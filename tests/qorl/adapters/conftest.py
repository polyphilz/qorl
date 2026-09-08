from pathlib import Path
from typing import Protocol

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from qorl.adapters.export import export_configuration
from qorl.adapters.merge import TENSOR_FILES
from qorl.adapters.schemas import AdapterExportManifest, LoraSettings
from qorl.model.files import model_weights_sha256
from qorl.util.hashing import sha256_file

WEIGHT = "model.layers.0.self_attn.q_proj.weight"
A_KEY = WEIGHT.removesuffix(".weight") + ".lora_A.weight"
B_KEY = WEIGHT.removesuffix(".weight") + ".lora_B.weight"


class ModelSaver(Protocol):
    def save_pretrained(
        self, save_directory: Path, *, max_shard_size: str | int
    ) -> None: ...


class TokenizerSaver(Protocol):
    def save_pretrained(self, save_directory: Path) -> tuple[str, ...]: ...


def save_model_and_tokenizer(
    model: ModelSaver, tokenizer: TokenizerSaver, base: Path, sharded: bool
) -> None:
    model.save_pretrained(base, max_shard_size=100 if sharded else "1GB")
    tokenizer.save_pretrained(base)


def record_adapter(base: Path, adapter: Path) -> None:
    tensors = TENSOR_FILES.load_file(adapter / "adapter_model.safetensors")
    manifest = AdapterExportManifest(
        tensor_count=len(tensors),
        nonzero_lora_b_values=1,
        adapter_sha256=sha256_file(adapter / "adapter_model.safetensors"),
        base_model_sha256=model_weights_sha256(base),
        checkpoint_sha256="test-checkpoint",
    )
    (adapter / "qorl-manifest.json").write_text(manifest.model_dump_json())


@pytest.fixture(params=[False, True], ids=["single", "sharded"])
def merge_inputs(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, Path]:
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "one": 1, "two": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    save_model_and_tokenizer(model, tokenizer, base, request.param is True)
    adapter.mkdir()
    tensors = {
        A_KEY: torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10,
        B_KEY: torch.ones((4, 2), dtype=torch.float32),
    }
    TENSOR_FILES.save_file(
        tensors, adapter / "adapter_model.safetensors", metadata={"format": "pt"}
    )
    config = export_configuration(
        base,
        LoraSettings(rank=2, alpha=6, dropout=0.2, target_modules=["q_proj"]),
        tensors,
    )
    (adapter / "adapter_config.json").write_text(config.model_dump_json())
    record_adapter(base, adapter)
    return base, adapter
