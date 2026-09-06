from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from qorl.adapters.schemas import AdapterConfig

ADAPTER_CONFIG_FILE = "adapter_config.json"


def adapter_config(adapter: Path) -> AdapterConfig:
    path = adapter / ADAPTER_CONFIG_FILE
    if not path.is_file():
        raise RuntimeError(f"adapter config is missing: {path}")
    try:
        return AdapterConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise RuntimeError(f"adapter config is invalid: {path}") from error


def adapter_rank(adapter: Path) -> int:
    return adapter_config(adapter).r
