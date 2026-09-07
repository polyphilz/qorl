"""Training settings shared by SFT and RL: runtime, LoRA, optimizer, and checkpoints."""

from typing import Literal, Self

from prime_rl.configs.trainer import AttnImplementation, SchedulerConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingRuntimeSettings(BaseModel):
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
                "training.runtime.attention requires implementation='custom' or 'auto'"
            )
        return self


class OptimizerSettings(BaseModel):
    """AdamW hyperparameters and scheduling, separate from gradient clipping."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    type: Literal["adamw"] = "adamw"
    lr: float = Field(default=1e-6, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    betas1: float = Field(default=0.9, ge=0)
    betas2: float = Field(default=0.999, ge=0)
    scheduler: SchedulerConfig


class CheckpointSettings(BaseModel):
    """An absent interval means final-only saving, not automatic evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["final", "interval"]
    interval: int | None = Field(default=None, ge=1)
    keep_last: int | None = Field(default=None, ge=1)
    keep_interval: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def applicable_schedule(self) -> Self:
        """Final-only saving has no interval or retention policy."""
        if self.type == "final" and any(
            value is not None
            for value in (self.interval, self.keep_last, self.keep_interval)
        ):
            raise ValueError(
                "final-only checkpoints do not use interval or retention settings"
            )
        if self.type == "interval" and self.interval is None:
            raise ValueError("interval checkpoints require an interval")
        return self
