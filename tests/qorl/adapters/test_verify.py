from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.adapters.verify import verify_merged_model
from qorl.util.hashing import sha256_file


def merged_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    for directory in (base, adapter, merged):
        directory.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}")
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
