from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.adapters.model import adapter_base_model, adapter_rank, verify_adapter_base
from qorl.util.hashing import sha256_file


def write_adapter_config(adapter: Path, base: Path, rank: object = 32) -> None:
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "bias": "none",
                "r": rank,
                "lora_alpha": 32.0,
            }
        )
    )


def test_adapter_rank_reads_the_adapter_config(tmp_path: Path) -> None:
    write_adapter_config(tmp_path, tmp_path)

    assert adapter_rank(tmp_path) == 32


@pytest.mark.parametrize("rank", [True, 0, "16", None])
def test_adapter_rank_rejects_invalid_values(tmp_path: Path, rank: object) -> None:
    write_adapter_config(tmp_path, tmp_path, rank)

    with pytest.raises(RuntimeError, match="adapter config is invalid"):
        adapter_rank(tmp_path)


def test_adapter_base_is_resolved_from_its_config(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    write_adapter_config(adapter, base)

    assert adapter_base_model(adapter, tmp_path) == base.resolve()
    assert verify_adapter_base(adapter, base, tmp_path) == sha256_file(
        base / "model.safetensors"
    )


def test_relative_adapter_base_is_resolved_from_the_repository(tmp_path: Path) -> None:
    base = tmp_path / "models/base"
    adapter = tmp_path / "adapter"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"base")
    write_adapter_config(adapter, Path("models/base"))

    assert adapter_base_model(adapter, tmp_path) == base.resolve()


def test_adapter_base_rejects_different_weights(tmp_path: Path) -> None:
    recorded = tmp_path / "recorded"
    supplied = tmp_path / "supplied"
    adapter = tmp_path / "adapter"
    recorded.mkdir()
    supplied.mkdir()
    (recorded / "model.safetensors").write_bytes(b"recorded")
    (supplied / "model.safetensors").write_bytes(b"supplied")
    write_adapter_config(adapter, recorded)

    with pytest.raises(
        RuntimeError, match="does not match the adapter's training base"
    ):
        verify_adapter_base(adapter, supplied, tmp_path)


def test_adapter_manifest_must_match_its_recorded_base(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    write_adapter_config(adapter, base)
    (adapter / "qorl-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensor_count": 1,
                "nonzero_lora_b_values": 1,
                "adapter_sha256": "adapter",
                "base_model_sha256": "wrong",
            }
        )
    )

    with pytest.raises(RuntimeError, match="does not match its recorded base"):
        verify_adapter_base(adapter, base, tmp_path)
