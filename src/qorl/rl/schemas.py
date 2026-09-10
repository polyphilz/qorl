"""RL batch settings and the parameters consumed by each learning algorithm."""

from pathlib import Path
from typing import Annotated, Literal, Self

import verifiers.v1 as vf
from prime_rl.configs.trainer import validate_scheduler
from pydantic import BaseModel, ConfigDict, Field, model_validator
from renderers import AutoRendererConfig, RendererConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
from verifiers.v1.episode import GroupInfo, TrainRunInfo
from verifiers.v1.trace import Error, TraceTask

from qorl.adapters.schemas import LoraSettings
from qorl.agent.schemas import AgentSettings, AgentTrace
from qorl.evaluation.schemas import PerformanceSummary
from qorl.measure.schemas import OutcomeKind, RolloutMeasurementSettings, RolloutRecord
from qorl.model.schemas import LocalInferenceSettings, ModelSettings, TokenUsage
from qorl.paths import REPOSITORY_ROOT
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
    renderer: RendererConfig = AutoRendererConfig()

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


class RlTrainingIdentity(BaseModel):
    """The recorded base and native trainer used by explicit checkpoint export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_sha256: str
    base_weights_sha256: str
    trainer_source: str
    trainer_config_sha256: str


class QorlTasksetConfig(vf.TasksetConfig):
    repository: Path = REPOSITORY_ROOT
    selection: Path | None = None
    shuffle_seed: int | None = None


class QorlTaskData(vf.TaskData):
    task_id: str
    template_id: str
    rollout_index: int = 0


class QorlHarnessConfig(vf.HarnessConfig):
    """Allow Verifiers' fallback construction; experiments supply resolved settings."""

    id: str = "qorl"
    seed: int = 42
    model: ModelSettings | None = None
    inference: LocalInferenceSettings | None = None
    agent: AgentSettings = AgentSettings(
        candidate_attempts=1,
        maximum_model_turns=64,
        inspection_turns_per_alias=3,
    )
    measurement: RolloutMeasurementSettings = RolloutMeasurementSettings(
        default_warmups=1,
        default_measurements=3,
        candidate_feedback_warmups=1,
        candidate_feedback_measurements=1,
        paired_warmups=1,
        paired_measurements=3,
        default_timeout_seconds=300.0,
        candidate_timeout_floor_seconds=5.0,
        candidate_timeout_multiplier=3.0,
    )
    rl: RlSettings = RlSettings(
        algorithm=AnchoredGrpoSettings(
            type="qorl_anchored_grpo",
            tau=0.05,
            c=0.10,
            d=0.02,
            t=0.10,
            min_peers=2,
        )
    )


class QorlEnvironmentConfig(SingleAgentEnvConfig):
    postgres_config: Path | None = None
    pool_config: Path | None = None


class AnchoredCredit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    discarded: bool
    discard_reason: str | None = None
    quality: float | None = None
    reference: float | None = None
    protocol_cost: float
    advantage: float


class ShipInfo(BaseModel):
    step: int


class TraceInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    qorl: RlRolloutRecord | None = None
    qorl_policy: AgentTrace | None = None
    qorl_advantage: AnchoredCredit | None = None
    ship: ShipInfo | None = None


class TraceEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    info: TraceInfo


class EpisodeEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    group: GroupInfo
    run: TrainRunInfo
    traces: list[TraceEvidence]
    task: TraceTask[QorlTaskData]
    ok: bool
    errors: list[Error] = []


class BranchCredit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    advantages: list[float] | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    trace_id: str
    info: TraceInfo
    branches: list[BranchCredit] = []


class LearningEvidence(BaseModel):
    """Join keys into native episodes, ship annotations, metrics and checkpoints."""

    episode_id: str
    group_id: str
    trace_id: str
    task_id: str | None
    policy_start: int | None
    policy_end: int | None
    ship_step: int | None
    anchored: AnchoredCredit | None
    assigned_advantage: float | None


class EpisodeFailure(BaseModel):
    episode_id: str
    group_id: str
    task_id: str
    errors: list[Error]


class UpdateMetric(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    producer: str | None = None
    step: int | None = None
    learning_rate: float | None = Field(default=None, alias="optim/lr")
    gradient_norm: float | None = Field(default=None, alias="optim/grad_norm")


class RlTrainingReport(BaseModel):
    """Usage sums available policy traces, including unscored rollouts.

    Missing policy usage counts traces; no-trace failures are counted separately
    in episode_failures. Provider-unknown token counts remain unknown.
    """

    schema_version: int = 1
    completed: bool
    scalar_reward_hook: str
    optimizer_steps: list[int]
    episode_count: int
    episode_failure_count: int
    episode_failures: list[EpisodeFailure]
    outcome_counts: dict[OutcomeKind, int]
    outcome_rates: dict[OutcomeKind, float | None]
    performance: PerformanceSummary
    learning: list[LearningEvidence]
    checkpoints: list[Path]
    usage: TokenUsage = TokenUsage()
    policy_usage_missing_count: int = 0
