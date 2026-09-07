"""CPU checkpoint export retains base provenance for each supported weight layout."""

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save as serialize_safetensors
from torch.distributed.checkpoint import state_dict_saver

from qorl.adapters import export
from qorl.adapters.schemas import AdapterExportManifest
from qorl.adapters.verify import verify_adapter_base
from qorl.model.files import model_weights_sha256
from qorl.model.schemas import ModelWeightIndex

WEIGHT_WIDTH = 2
LORA_RANK = 1


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("sharded", [False, True])
def test_export_records_verifiable_base_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    sharded: bool,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    weights = {"layer.weight": torch.eye(WEIGHT_WIDTH)}
    weight_file = base / (f"shard-{filename}" if sharded else filename)
    if filename.endswith("safetensors"):
        weight_file.write_bytes(serialize_safetensors(weights))
    else:
        torch.save(weights, weight_file)
    if sharded:
        (base / f"{filename}.index.json").write_text(
            ModelWeightIndex(
                weight_map={
                    "layer.weight": weight_file.name,
                }
            ).model_dump_json()
        )
    checkpoint = tmp_path / "checkpoint"
    state_dict_saver.save(
        {
            "app": {
                "model": {
                    "layer.lora_A.0": torch.ones((LORA_RANK, WEIGHT_WIDTH)),
                    "layer.lora_B.0": torch.ones((WEIGHT_WIDTH, LORA_RANK)),
                }
            }
        },
        checkpoint_id=checkpoint,
    )
    output = tmp_path / "adapter"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--model",
            str(base),
            "--output",
            str(output),
        ],
    )
    export.main()
    manifest = AdapterExportManifest.model_validate_json(
        (output / "qorl-manifest.json").read_bytes()
    )
    expected = model_weights_sha256(base)
    assert manifest.base_model_sha256 == expected
    assert verify_adapter_base(output, base) == expected
