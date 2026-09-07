"""RL batch settings and the parameters consumed by each learning algorithm."""

from typing import Annotated, Literal, Self

from prime_rl.configs.trainer import validate_scheduler
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qorl.adapters.schemas import LoraSettings
from qorl.measure.schemas import RolloutRecord
from qorl.training.schemas import (
    CheckpointSettings,
    OptimizerSettings,
    TrainingRuntimeSettings,
)
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest


class RlTrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(ge=1)
    group_size: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    max_off_policy_steps: int = Field(ge=0)
    max_inflight: int = Field(ge=1)
    max_grad_norm: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Maximum gradient norm; omission disables clipping, zero does not.",
    )
    runtime: TrainingRuntimeSettings
    lora: LoraSettings
    optimizer: OptimizerSettings
    checkpoints: CheckpointSettings

    @model_validator(mode="after")
    def whole_groups(self) -> Self:
        """Batches contain whole groups, and concurrency can accommodate one group."""
        if self.batch_size % self.group_size:
            raise ValueError("training.batch_size must be divisible by group_size")
        if self.max_inflight < self.group_size:
            raise ValueError("training.max_inflight must be at least group_size")
        return self

    @model_validator(mode="after")
    def learning_rate_schedule(self) -> Self:
        """Check scheduler phases against the configured optimizer step count."""
        validate_scheduler(self.optimizer.scheduler, self.max_steps)
        return self


class AnchoredGrpoSettings(BaseModel):
    """expected_group_size is derived from training.group_size at translation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    type: Literal["qorl_anchored_grpo"]
    tau: float = Field(ge=0)
    c: float = Field(gt=0)
    d: float = Field(ge=0)
    t: float = Field(ge=0)
    min_peers: int = Field(ge=1)


class GrpoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["grpo"]


class ScalarRewardSettings(BaseModel):
    """Per-attempt scalar penalties; anchored GRPO does not consume these."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    invalid_attempt_penalty: float = Field(ge=0)
    duplicate_attempt_penalty: float = Field(ge=0)
    timeout_attempt_penalty: float = Field(ge=0)
    no_valid_candidate_reward: float


class RlSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Annotated[
        AnchoredGrpoSettings | GrpoSettings, Field(discriminator="type")
    ]
    reward: ScalarRewardSettings | None = None

    @model_validator(mode="after")
    def consumed_reward_settings(self) -> Self:
        """Do not expose scalar penalties that the selected algorithm ignores."""
        if isinstance(self.algorithm, AnchoredGrpoSettings) and self.reward is not None:
            raise ValueError("anchored GRPO does not consume rl.reward")
        if isinstance(self.algorithm, GrpoSettings) and self.reward is None:
            raise ValueError("GRPO requires rl.reward")
        return self


class RlRolloutRecord(RolloutRecord):
    """Measurement facts plus runtime scope and optional ordinary-GRPO scalar reward."""

    database_pool: PoolManifest
    database_worker: WorkerManifest
    scalar_reward: float | None
