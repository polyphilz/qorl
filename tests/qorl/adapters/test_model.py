from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorl.adapters.model import adapter_rank


def test_adapter_rank_reads_the_adapter_config(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32}))

    assert adapter_rank(tmp_path) == 32


@pytest.mark.parametrize("rank", [True, 0, "16", None])
def test_adapter_rank_rejects_invalid_values(tmp_path: Path, rank: object) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": rank}))

    with pytest.raises(RuntimeError, match="invalid LoRA rank"):
        adapter_rank(tmp_path)
