from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoraSettings(BaseModel):
    """LoRA parameters shared by training and adapter export."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    rank: int = Field(ge=1)
    alpha: float = Field(ge=0)
    dropout: float = Field(ge=0, le=1)
    target_modules: list[str] = Field(min_length=1)


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
    checkpoint_sha256: str | None = None


class MergeLoraConfig(BaseModel):
    """The plain, uniform-rank LoRA format supported by offline merging."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    base_model_name_or_path: str = Field(min_length=1)
    revision: str | None = None
    peft_type: Literal["LORA"]
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    bias: Literal["none"]
    r: int = Field(gt=0)
    lora_alpha: float = Field(ge=0)
    lora_dropout: float = Field(default=0, ge=0, le=1)
    target_modules: list[str] = Field(min_length=1)
    modules_to_save: None = None
    fan_in_fan_out: Literal[False] = False
    use_rslora: Literal[False] = False
    use_dora: Literal[False] = False
    lora_bias: Literal[False] = False
    rank_pattern: dict[str, int] = Field(default_factory=dict, max_length=0)
    alpha_pattern: dict[str, float] = Field(default_factory=dict, max_length=0)


class MergeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    bytes: int = Field(ge=0)
    sha256: str


class TensorSignature(BaseModel):
    shape: list[int]
    dtype: str


class MergeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    operation: Literal["merge_lora_into_base"] = "merge_lora_into_base"
    base_model_sha256: str
    adapter_model_sha256: str
    adapter_config_sha256: str
    adapter_manifest_sha256: str
    merged_model_sha256: str
    lora_rank: int
    lora_alpha: float
    lora_scale: float
    merged_tensor_count: int
    artifacts: list[MergeArtifact]
