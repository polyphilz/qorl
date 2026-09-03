from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def model_snapshot(policy: dict[str, Any]) -> Path:
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache = str(home / "hub")
    snapshot = (
        Path(cache)
        / f"models--{policy['model'].replace('/', '--')}"
        / "snapshots"
        / policy["revision"]
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned model snapshot is missing: {snapshot}")
    return snapshot.resolve()


def adapter_rank(adapter: Path) -> int:
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise RuntimeError(f"adapter has an invalid LoRA rank: {config_path}")
    return rank
