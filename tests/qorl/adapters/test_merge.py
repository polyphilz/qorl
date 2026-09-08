import json
from pathlib import Path
from typing import Protocol
from unittest.mock import Mock

import pytest
import torch
from tests.qorl.adapters.conftest import A_KEY, B_KEY, WEIGHT, record_adapter
from transformers import (
    LlamaForCausalLM,
    PreTrainedModel,
)

from qorl.adapters import merge as merging
from qorl.adapters.merge import TENSOR_FILES, merge
from qorl.adapters.schemas import MergeManifest
from qorl.adapters.verify import (
    TOKENIZER_LOADER,
    artifact_inventory,
    verify_merged_model,
)
from qorl.model.files import model_weight_files, model_weights_sha256
from qorl.model.schemas import JsonValue
from qorl.util.io import write_json


class ModelLoader(Protocol):
    @classmethod
    def from_pretrained(
        cls, path: Path, /, *, local_files_only: bool
    ) -> PreTrainedModel: ...


MODEL_LOADER: type[ModelLoader] = LlamaForCausalLM


def test_merge_arithmetic_layout_and_loading(
    merge_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    base, adapter = merge_inputs
    before = model_weights_sha256(base)
    (base / "qorl-merge.json").write_text('{"stale": true}')
    (base / "pytorch_model.bin").write_bytes(b"stale competing weights")
    output = merge(base, adapter, tmp_path / "output")
    assert output == tmp_path / "output"
    assert model_weights_sha256(base) == before
    original = MODEL_LOADER.from_pretrained(base, local_files_only=True)
    merged = MODEL_LOADER.from_pretrained(output, local_files_only=True)
    tokenizer = TOKENIZER_LOADER.from_pretrained(output, local_files_only=True)
    assert tokenizer.get_vocab() == {"[UNK]": 0, "one": 1, "two": 2}
    updates = TENSOR_FILES.load_file(adapter / "adapter_model.safetensors")
    for key, value in original.state_dict().items():
        expected = (
            value + 3 * (updates[B_KEY] @ updates[A_KEY]) if key == WEIGHT else value
        )
        torch.testing.assert_close(merged.state_dict()[key], expected)
    assert [p.name for p in model_weight_files(base)] == [
        p.name for p in model_weight_files(output)
    ]
    assert not (output / "pytorch_model.bin").exists()
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        assert (output / filename).read_bytes() == (base / filename).read_bytes()
    manifest = MergeManifest.model_validate_json(
        (output / "qorl-merge.json").read_bytes()
    )
    assert (
        manifest.lora_rank == 2
        and manifest.lora_alpha == 6
        and manifest.lora_scale == 3
    )
    assert manifest.merged_tensor_count == 1
    assert manifest.artifacts == artifact_inventory(output)
    verify_merged_model(base, adapter, output)
    (output / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="artifact inventory differs"):
        verify_merged_model(base, adapter, output)


def test_identical_relocated_base(
    merge_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    base, adapter = merge_inputs
    relocated = tmp_path / "relocated"
    base.rename(relocated)
    output = merge(relocated, adapter, tmp_path / "output")
    verify_merged_model(relocated, adapter, output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_rslora", True),
        ("use_dora", True),
        ("fan_in_fan_out", True),
        ("bias", "all"),
        ("modules_to_save", ["lm_head"]),
        ("rank_pattern", {"q_proj": 1}),
        ("alpha_pattern", {"q_proj": 1.0}),
        ("lora_bias", True),
        ("lora_alpha", -1),
        ("unknown_feature", True),
    ],
)
def test_reject_unsupported_settings(
    merge_inputs: tuple[Path, Path], tmp_path: Path, field: str, value: JsonValue
) -> None:
    base, adapter = merge_inputs
    path = adapter / "adapter_config.json"
    config = merging.OBJECT.validate_json(path.read_bytes())
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change",
    ["rank", "dimension", "missing_b", "unknown_key", "wrong_target", "integer", "nan"],
)
def test_reject_tensor_mismatches(
    merge_inputs: tuple[Path, Path], tmp_path: Path, change: str
) -> None:
    base, adapter = merge_inputs
    path = adapter / "adapter_model.safetensors"
    tensors = TENSOR_FILES.load_file(path)
    if change == "rank":
        tensors[A_KEY] = torch.ones((1, 4))
    elif change == "dimension":
        tensors[B_KEY] = torch.ones((4, 2, 1))
    elif change == "missing_b":
        del tensors[B_KEY]
    elif change == "unknown_key":
        tensors["extra"] = torch.ones(1)
    elif change == "wrong_target":
        tensors = {
            key.replace("q_proj", "invented"): value for key, value in tensors.items()
        }
    elif change == "integer":
        tensors[A_KEY] = tensors[A_KEY].int()
    else:
        tensors[A_KEY][0, 0] = float("nan")
    TENSOR_FILES.save_file(tensors, path, metadata={"format": "pt"})
    record_adapter(base, adapter)
    with pytest.raises(RuntimeError):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("corrupt", ["base", "adapter"])
def test_reject_corrupt_weights(
    merge_inputs: tuple[Path, Path], tmp_path: Path, corrupt: str
) -> None:
    base, adapter = merge_inputs
    path = (
        model_weight_files(base)[-1]
        if corrupt == "base"
        else adapter / "adapter_model.safetensors"
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="manifest"):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_staging_failure_and_no_overwrite(
    merge_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter = merge_inputs
    original = artifact_inventory(base), artifact_inventory(adapter)
    output = tmp_path / "output"
    monkeypatch.setattr(
        merging,
        "verify_merged_model",
        Mock(side_effect=RuntimeError("staging rejected")),
    )
    with pytest.raises(RuntimeError, match="staging rejected"):
        merge(base, adapter, output)
    assert not output.exists() and not list(tmp_path.glob(".output-*"))
    output.mkdir()
    (output / "user-file").write_text("preserve")
    with pytest.raises(RuntimeError, match="already exists"):
        merge(base, adapter, output)
    assert (output / "user-file").read_text() == "preserve"
    assert (artifact_inventory(base), artifact_inventory(adapter)) == original


@pytest.mark.parametrize("source", ["base", "adapter"])
def test_output_within_source(
    merge_inputs: tuple[Path, Path], tmp_path: Path, source: str
) -> None:
    base, adapter = merge_inputs
    parent = base if source == "base" else adapter
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    original = artifact_inventory(parent)
    for path in (parent / "output", alias / "nested" / "output"):
        with pytest.raises(RuntimeError, match="outside"):
            merge(base, adapter, path)
    assert artifact_inventory(parent) == original
    assert not list(parent.glob(".*")) and not (parent / "nested").exists()


def test_incomplete_tokenizer_is_not_published(
    merge_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    base, adapter = merge_inputs
    (base / "tokenizer.json").unlink()
    with pytest.raises((OSError, ValueError, TypeError)):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize(
    "field", ["lora_rank", "lora_alpha", "lora_scale", "merged_tensor_count"]
)
def test_verify_rejects_changed_recorded_facts(
    merge_inputs: tuple[Path, Path], tmp_path: Path, field: str
) -> None:
    base, adapter = merge_inputs
    output = merge(base, adapter, tmp_path / "output")
    path = output / "qorl-merge.json"
    manifest = merging.OBJECT.validate_json(path.read_bytes())
    manifest[field] = 123
    write_json(path, manifest)
    with pytest.raises(RuntimeError, match="scaling or tensor count"):
        verify_merged_model(base, adapter, output)


@pytest.mark.parametrize("shape", [(3, 5, 1), (5, 2, 2), (2, 4, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 7.0])
def test_non_square_arithmetic(
    shape: tuple[int, int, int], dtype: torch.dtype, alpha: float
) -> None:
    rows, columns, rank = shape
    weight = torch.ones((rows, columns), dtype=dtype)
    a = (
        torch.arange(rank * columns, dtype=torch.float32)
        .reshape(rank, columns)
        .to(dtype)
    )
    b = torch.ones((rows, rank), dtype=dtype)
    config = merging.MergeLoraConfig(
        base_model_name_or_path="base",
        peft_type="LORA",
        bias="none",
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj"],
    )
    actual = merging.merged_weight(WEIGHT, weight, a, b, config)
    expected = (weight.float() + alpha / rank * (b.float() @ a.float())).to(dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    assert torch.equal(weight, torch.ones_like(weight))


def test_staged_tensor_validation(
    merge_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter = merge_inputs

    class WrongShapeWriter:
        def load_file(self, filename: Path) -> dict[str, torch.Tensor]:
            return TENSOR_FILES.load_file(filename)

        def save_file(
            self,
            tensors: dict[str, torch.Tensor],
            filename: Path,
            metadata: dict[str, str],
        ) -> None:
            if WEIGHT in tensors:
                tensors[WEIGHT] = torch.ones(1)
            TENSOR_FILES.save_file(tensors, filename, metadata)

    monkeypatch.setattr(merging, "TENSOR_FILES", WrongShapeWriter())
    with pytest.raises(RuntimeError, match="tensor names, shapes or dtypes"):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))


def test_output_appearing_during_staging_is_preserved(
    merge_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter = merge_inputs
    output = tmp_path / "output"

    def verify_and_occupy(base: Path, adapter: Path, staging: Path) -> None:
        verify_merged_model(base, adapter, staging)
        output.mkdir()

    monkeypatch.setattr(merging, "verify_merged_model", verify_and_occupy)
    with pytest.raises(RuntimeError, match="already exists"):
        merge(base, adapter, output)
    assert output.is_dir() and list(output.iterdir()) == []
    assert not list(tmp_path.glob(".output-*"))


def test_inconsistent_shard_index_is_rejected(
    merge_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    base, adapter = merge_inputs
    index = base / "model.safetensors.index.json"
    if not index.exists():
        index.write_text(json.dumps({"weight_map": {"invented": "model.safetensors"}}))
    else:
        document = merging.OBJECT.validate_json(index.read_bytes())
        document["weight_map"] = {"invented": model_weight_files(base)[-1].name}
        write_json(index, document)
    record_adapter(base, adapter)
    with pytest.raises(RuntimeError, match="index does not match"):
        merge(base, adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()
