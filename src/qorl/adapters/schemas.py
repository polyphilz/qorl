from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    base_model_name_or_path: str = Field(min_length=1)
    peft_type: str
    bias: str
    r: int = Field(gt=0)
    lora_alpha: float


class AdapterExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    tensor_count: int = Field(gt=0)
    nonzero_lora_b_values: int = Field(gt=0)
    adapter_sha256: str
    base_model_sha256: str | None = None
