"""Model identities and local decoding settings owned by an experiment."""

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from prime_rl.configs.trainer import AttnImplementation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ModelProvider
    name_or_path: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    revision: str | None = None
    adapter_path: Path | None = None


class LocalDecodingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0, le=1)
    presence_penalty: float
    repetition_penalty: float = Field(gt=0)
    thinking: bool


class ModelWeightIndex(BaseModel):
    """Shard filenames from a Hugging Face model's weights index."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    weight_map: dict[str, str] = Field(min_length=1)


class TrainerModelSettings(BaseModel):
    """Weight-update implementation settings, separate from model identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    implementation: Literal["hf", "custom", "auto"]
    attention: AttnImplementation
    optimization_dtype: Literal["bfloat16", "float32"]
    reduce_dtype: Literal["bfloat16", "float32"]
    compile: bool

    @model_validator(mode="after")
    def supported_attention(self) -> Self:
        """FA4, including automatic selection, requires Prime-RL's implementation."""
        if (
            self.attention in ("auto", "flash_attention_4")
            and self.implementation == "hf"
        ):
            raise ValueError(
                "training.model.attention requires implementation='custom' or 'auto'"
            )
        return self
