from __future__ import annotations

import os
from pathlib import Path


def model_snapshot(model: str, revision: str) -> Path:
    """Find an already-downloaded model revision in the Hugging Face cache."""
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache = str(home / "hub")
    snapshot = (
        Path(cache) / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned model snapshot is missing: {snapshot}")
    return snapshot.resolve()
