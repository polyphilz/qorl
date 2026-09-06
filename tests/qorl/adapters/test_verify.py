from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.adapters import verify
from qorl.adapters.verify import verify_adapter_base, verify_merged_model
from qorl.util.hashing import sha256_file


def merged_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    for directory in (base, adapter, merged):
        directory.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "bias": "none",
                "r": 16,
                "lora_alpha": 32.0,
            }
        )
    )
    (merged / "model.safetensors").write_bytes(b"merged")
    (merged / "tokenizer.json").write_bytes(b"tokenizer")
    artifacts = [
        {
            "path": path.relative_to(merged).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(merged.iterdir())
    ]
    (merged / "qorl-merge.json").write_text(
        json.dumps(
            {
                "base_model_sha256": sha256_file(base / "model.safetensors"),
                "adapter_model_sha256": sha256_file(
                    adapter / "adapter_model.safetensors"
                ),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
                "merged_model_sha256": sha256_file(merged / "model.safetensors"),
                "artifacts": artifacts,
            }
        )
    )
    return base, adapter, merged


def test_merged_model_verifies_every_recorded_artifact(tmp_path: Path) -> None:
    base, adapter, merged = merged_fixture(tmp_path)

    verify_merged_model(base, adapter, merged)

    (merged / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact inventory differs"):
        verify_merged_model(base, adapter, merged)


def test_merged_model_rejects_an_adapter_applied_to_the_wrong_base(
    tmp_path: Path,
) -> None:
    _, adapter, merged = merged_fixture(tmp_path)
    wrong_base = tmp_path / "wrong-base"
    wrong_base.mkdir()
    (wrong_base / "model.safetensors").write_bytes(b"wrong")

    with pytest.raises(
        RuntimeError, match="does not match the adapter's training base"
    ):
        verify_merged_model(wrong_base, adapter, merged)


def test_adapter_base_is_resolved_from_its_config(tmp_path: Path) -> None:
    base, adapter, _ = merged_fixture(tmp_path)

    assert verify_adapter_base(adapter, base) == sha256_file(base / "model.safetensors")


def test_relative_adapter_base_is_resolved_from_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter, _ = merged_fixture(tmp_path)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "base"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(verify, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(adapter)

    assert verify_adapter_base(adapter, base) == sha256_file(base / "model.safetensors")


def test_adapter_base_rejects_different_weights(tmp_path: Path) -> None:
    _, adapter, _ = merged_fixture(tmp_path)
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    (supplied / "model.safetensors").write_bytes(b"supplied")

    with pytest.raises(
        RuntimeError, match="does not match the adapter's training base"
    ):
        verify_adapter_base(adapter, supplied)


def test_adapter_manifest_must_match_its_recorded_base(tmp_path: Path) -> None:
    base, adapter, _ = merged_fixture(tmp_path)
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
        verify_adapter_base(adapter, base)


def test_adapter_base_requires_recorded_weights(tmp_path: Path) -> None:
    base, adapter, _ = merged_fixture(tmp_path)
    (base / "model.safetensors").unlink()

    with pytest.raises(RuntimeError, match="adapter's recorded base model is missing"):
        verify_adapter_base(adapter, base)
