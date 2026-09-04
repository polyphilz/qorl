from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qorl.adapters.schemas import AdapterConfig, AdapterExportManifest
from qorl.util.hashing import sha256_file

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_MANIFEST_FILE = "qorl-manifest.json"
MODEL_FILE = "model.safetensors"


def adapter_config(adapter: Path) -> AdapterConfig:
    path = adapter / ADAPTER_CONFIG_FILE
    if not path.is_file():
        raise RuntimeError(f"adapter config is missing: {path}")
    try:
        return AdapterConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise RuntimeError(f"adapter config is invalid: {path}") from error


def adapter_base_model(adapter: Path, repository: Path) -> Path:
    recorded = Path(adapter_config(adapter).base_model_name_or_path).expanduser()
    base = (recorded if recorded.is_absolute() else repository / recorded).resolve()
    if not (base / MODEL_FILE).is_file():
        raise RuntimeError(f"adapter's recorded base model is missing: {base}")
    return base


def verify_adapter_base(adapter: Path, base: Path, repository: Path) -> str:
    recorded = adapter_base_model(adapter, repository)
    recorded_sha256 = sha256_file(recorded / MODEL_FILE)
    manifest_path = adapter / ADAPTER_MANIFEST_FILE
    if manifest_path.is_file():
        manifest = AdapterExportManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            manifest.base_model_sha256 is not None
            and manifest.base_model_sha256 != recorded_sha256
        ):
            raise RuntimeError(
                "adapter manifest does not match its recorded base model"
            )
    supplied_base = base.resolve()
    supplied = supplied_base / MODEL_FILE
    if not supplied.is_file():
        raise RuntimeError(f"base model weights are missing: {supplied}")
    supplied_sha256 = (
        recorded_sha256 if supplied_base == recorded else sha256_file(supplied)
    )
    if supplied_sha256 != recorded_sha256:
        raise RuntimeError(
            "supplied base model does not match the adapter's training base"
        )
    return recorded_sha256


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
    return adapter_config(adapter).r
