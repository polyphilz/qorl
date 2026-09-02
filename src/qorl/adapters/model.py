from __future__ import annotations

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
