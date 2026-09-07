"""RL batch settings and the parameters consumed by each learning algorithm."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qorl.model.schemas import TrainerModelSettings


class RlTrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(ge=1)
    group_size: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    max_off_policy_steps: int = Field(ge=0)
    max_inflight: int = Field(ge=1)
    model: TrainerModelSettings

    @model_validator(mode="after")
    def whole_groups(self) -> Self:
        """Batches contain whole groups, and concurrency can accommodate one group."""
        if self.batch_size % self.group_size:
            raise ValueError("training.batch_size must be divisible by group_size")
        if self.max_inflight < self.group_size:
            raise ValueError("training.max_inflight must be at least group_size")
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
