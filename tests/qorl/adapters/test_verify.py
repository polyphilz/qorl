from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qorl.adapters import verify
from qorl.adapters.schemas import AdapterExportManifest
from qorl.adapters.verify import verify_adapter_base, verify_merged_model
from qorl.model.files import model_weights_sha256
from qorl.model.schemas import ModelWeightIndex
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


@pytest.fixture(params=["model.safetensors", "pytorch_model.bin"])
def sharded_adapter(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[Path, Path, Path]:
    filename = str(request.param)
    base, adapter, _ = merged_fixture(tmp_path)
    (base / "model.safetensors").unlink()
    (base / f"first-{filename}").write_bytes(b"first shard")
    (base / f"second-{filename}").write_bytes(b"second shard")
    (base / f"{filename}.index.json").write_text(
        ModelWeightIndex(
            weight_map={
                "a": f"first-{filename}",
                "b": f"second-{filename}",
            }
        ).model_dump_json()
    )
    (adapter / "qorl-manifest.json").write_text(
        AdapterExportManifest(
            tensor_count=1,
            nonzero_lora_b_values=1,
            adapter_sha256=sha256_file(adapter / "adapter_model.safetensors"),
            base_model_sha256=model_weights_sha256(base),
        ).model_dump_json()
    )
    supplied = tmp_path / "supplied"
    shutil.copytree(base, supplied)
    return base, adapter, supplied


def test_sharded_adapter_accepts_original_or_identical_copied_base(
    sharded_adapter: tuple[Path, Path, Path],
) -> None:
    base, adapter, supplied = sharded_adapter
    expected = model_weights_sha256(base)
    assert verify_adapter_base(adapter, base) == expected
    assert verify_adapter_base(adapter, supplied) == expected


@pytest.mark.parametrize("file_index", [0, 1, 2])
def test_sharded_adapter_rejects_modified_supplied_weights(
    sharded_adapter: tuple[Path, Path, Path], file_index: int
) -> None:
    _, adapter, supplied = sharded_adapter
    files = sorted(supplied.iterdir())
    path = files[file_index]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(
        RuntimeError, match="does not match the adapter's training base"
    ):
        verify_adapter_base(adapter, supplied)


def test_sharded_adapter_manifest_detects_changes_at_the_recorded_path(
    sharded_adapter: tuple[Path, Path, Path],
) -> None:
    base, adapter, _ = sharded_adapter
    first = next(path for path in base.iterdir() if path.name.startswith("first-"))
    first.write_bytes(b"changed after export")
    with pytest.raises(RuntimeError, match="does not match its recorded base"):
        verify_adapter_base(adapter, base)


def test_sharded_adapter_rejects_incomplete_supplied_base(
    sharded_adapter: tuple[Path, Path, Path],
) -> None:
    _, adapter, supplied = sharded_adapter
    next(path for path in supplied.iterdir() if path.name.startswith("first-")).unlink()
    with pytest.raises(RuntimeError, match="missing or invalid"):
        verify_adapter_base(adapter, supplied)
