from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.adapters.config import adapter_config, adapter_rank


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


def test_adapter_config_requires_a_config_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="adapter config is missing"):
        adapter_config(tmp_path)
