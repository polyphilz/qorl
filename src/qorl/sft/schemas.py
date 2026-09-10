from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    TypeAdapter,
    model_validator,
)
from renderers.configs import AutoRendererConfig, RendererConfig

from qorl.adapters.schemas import LoraSettings
from qorl.agent.interface import AGENT_INTERFACE_VERSION
from qorl.agent.schemas import AgentSettings, AgentTrace
from qorl.measure.schemas import (
    Baseline,
    Candidate,
    RolloutMeasurementSettings,
    RolloutRecord,
    RunStatus,
    SelectionState,
)
from qorl.model.schemas import (
    OPENROUTER_MODEL_ID,
    AstraInferenceSettings,
    Message,
    ModelProvider,
    ModelSettings,
    OpenRouterInferenceSettings,
    ToolDefinition,
)
from qorl.plans.fingerprint import PLAN_FINGERPRINT_VERSION
from qorl.taskset.schemas import Task, TaskRole, TaskSelection, TaskSelectionInput
from qorl.training.schemas import (
    CheckpointSettings,
    OptimizerSettings,
    TrainingRuntimeSettings,
)
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest

type JsonObject = dict[str, JsonValue]

JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
GENERATOR_MODELS = {
    ModelProvider.OPENAI: "gpt-6-astra",
    ModelProvider.OPENROUTER: OPENROUTER_MODEL_ID,
}


class SftTrainingSettings(BaseModel):
    """Epochs and packed-row batch sizes, not original conversation counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=1)
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
    def whole_microbatches(self) -> Self:
        """An optimizer batch contains whole microbatches of packed rows."""
        if self.batch_size % self.micro_batch_size:
            raise ValueError(
                "training.batch_size must be divisible by micro_batch_size"
            )
        return self


class GenerationSettings(BaseModel):
    """Hosted teacher settings, independent of student rendering and decoding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelSettings | Literal["FILL_ME_IN"]
    generations_per_task: Annotated[int, Field(ge=1)] | Literal["FILL_ME_IN"]
    inference: AstraInferenceSettings | OpenRouterInferenceSettings
    plan_only: bool = False

    @model_validator(mode="after")
    def hosted_generator(self) -> Self:
        """Generation uses a hosted model, never the trainee or a local adapter."""
        if isinstance(self.model, ModelSettings):
            if self.model.provider == ModelProvider.LOCAL:
                raise ValueError("SFT generation requires a supported hosted teacher")
            if self.model.name_or_path != GENERATOR_MODELS[self.model.provider]:
                raise ValueError(
                    "SFT generation supports only GPT-6 Astra or the configured OpenRouter Qwen"
                )
            if self.model.provider == ModelProvider.OPENROUTER:
                if not isinstance(self.inference, OpenRouterInferenceSettings):
                    raise ValueError("OpenRouter teacher requires OpenRouter inference")
                self.inference.validate_model(self.model)
            elif not isinstance(self.inference, AstraInferenceSettings):
                raise ValueError("Astra teacher requires Astra inference")
            if self.model.revision is not None or self.model.adapter_path is not None:
                raise ValueError(
                    "hosted generators do not accept revisions or adapters"
                )
            if self.inference.max_tokens > self.model.context_length:
                raise ValueError(
                    "teacher inference.max_tokens exceeds teacher context_length"
                )
        return self


class PreparedDatasetSplit(BaseModel):
    """Frozen task IDs and original seeds accompanying one conversation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection: TaskSelection
    selection_seed: int
    generation_seed: int
    selection_expression: str | None = None
    conversations: Path


class PreparedDatasetManifest(BaseModel):
    """manifest.json in a reusable QORL conversation artifact, before token packing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2]
    format: Literal["qorl-conversations"]
    training: PreparedDatasetSplit
    validation: PreparedDatasetSplit


class ImportedGenerationSeeds(BaseModel):
    """Keep source generation seeds distinct from a new experiment's seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training: int
    validation: int


class ConversationRequest(BaseModel):
    """The exact ordered tool definitions supplied before one assistant reply."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assistant_message_index: int = Field(ge=0)
    tools: list[ToolDefinition]


class Conversation(BaseModel):
    """One frozen conversation; metadata belongs to its original generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    conversation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    requests: list[ConversationRequest]
    metadata: JsonObject


class TrainingRow(BaseModel):
    """Prime-RL's causally shifted, text-only sample or packed row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ids: list[int] = Field(min_length=1)
    target_ids: list[int]
    loss_mask: list[bool]
    position_ids: list[int]
    seq_lens: list[int]
    mm_kwargs: None = None
    mm_token_type_ids: None = None

    @model_validator(mode="after")
    def aligned_tokens(self) -> Self:
        """Every input has a target, mask and position; boundaries cover the row."""
        length = len(self.input_ids)
        if any(
            len(values) != length
            for values in (self.target_ids, self.loss_mask, self.position_ids)
        ):
            raise ValueError(
                "training row token, target, mask and position lengths differ"
            )
        if not self.seq_lens or min(self.seq_lens) < 1 or sum(self.seq_lens) != length:
            raise ValueError("training row seq_lens must cover the row exactly")
        return self


class RenderingRejection(StrEnum):
    OVER_CONTEXT = "over_context"
    NO_TARGETS = "no_targets"


class RenderedRequest(BaseModel):
    """One request's prefix and reply; only that reply contributes to loss."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    assistant_message_index: int
    tools_sha256: str
    rendered_text: str
    target_message_indices: list[int]
    sample: TrainingRow

    @model_validator(mode="after")
    def target_attribution(self) -> Self:
        """Each shifted target retains exactly one source-message index."""
        if len(self.target_message_indices) != len(self.sample.target_ids):
            raise ValueError("target_message_indices must align with target_ids")
        return self


class RecordedActionValidity(BaseModel):
    """Only explicit saved action validity determines semantic supervision eligibility."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    action_valid: StrictBool | None = None


class SkippedRequest(BaseModel):
    """An intact assistant reply excluded by its recorded schema or action feedback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assistant_message_index: int = Field(ge=0)
    reasons: list[str] = Field(min_length=1)


class RenderedConversation(BaseModel):
    """A conversation's request-level samples, accepted or rejected together."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    conversation_id: str
    task_id: str
    requests: list[RenderedRequest]
    skipped_requests: list[SkippedRequest] = Field(default_factory=list[SkippedRequest])
    rejection: RenderingRejection | None = None


class PackingRowMetadata(BaseModel):
    """Request order and unpadded lengths alongside native packed seq_lens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_ids: list[str]
    assistant_message_indices: list[int]
    request_lengths: list[int]
    padding_tokens: int = Field(ge=0)


class DatasetPreparationIdentity(BaseModel):
    """Inputs that must remain identical when resuming token preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    config_sha256: str
    source: Path
    source_files: dict[str, str]
    tokenizer_sha256: str
    renderer: JsonObject
    dependencies: dict[str, str]


class PreparedSplitReport(BaseModel):
    """Count tasks, conversations, accepted request samples and packed rows separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_tasks: int
    source_conversations: int
    accepted_conversations: int
    accepted_requests: int
    skipped_requests: int = Field(default=0, ge=0)
    accepted_tasks: int
    rejections: dict[RenderingRejection, int]
    packed_rows: int
    input_tokens: int
    supervised_tokens: int
    padding_tokens: int


class DatasetPreparationReport(BaseModel):
    """Completion marker with both split counts and checksums of prepared files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    context_length: int
    shuffle_seed: int
    training: PreparedSplitReport
    validation: PreparedSplitReport
    files: dict[str, str]


class SftContinuation(BaseModel):
    """Read-only completed checkpoint supplying optimizer state and absolute data position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: Path
    checkpoint_sha256: str
    completed_steps: int = Field(gt=0)
    historical_rng_restored: Literal[False] = False


class SftInitialization(BaseModel):
    """An exported adapter supplying weights for a fresh training schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: Path
    adapter_sha256: str
    config_sha256: str
    manifest_sha256: str


class TrainingIdentity(BaseModel):
    """Recorded training inputs and the base used to export its adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_sha256: str
    preparation_sha256: str
    base_weights_sha256: str
    trainer_source: str
    trainer_config_sha256: str
    continuation: SftContinuation | None = None
    initialization: SftInitialization | None = None


class TrainerMetric(BaseModel):
    """The fields consumed from Prime-RL's otherwise unchanged metrics JSONL."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    step: int
    validation_loss: float | None = Field(default=None, alias="val/loss")


class ValidationLoss(BaseModel):
    """Loss on validation targets at a completed optimizer-update count."""

    step: int
    loss: float = Field(allow_inf_nan=False)


class SftTrainingReport(BaseModel):
    """Completed schedule, dataset counts and explicitly identified saved weights."""

    schema_version: Literal[1] = 1
    training: PreparedSplitReport
    validation: PreparedSplitReport
    epochs: int
    batch_size: int
    steps_per_epoch: int
    optimizer_updates: int
    starting_step: int = Field(default=0, ge=0)
    validation_losses: list[ValidationLoss]
    checkpoints: list[Path]
    final_adapter: Path


def load_json_object(path: Path) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def load_json_lines[RecordT: BaseModel](
    path: Path, record: type[RecordT]
) -> list[RecordT]:
    return [
        record.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def require_object(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list")
    return value


class GenerationIdentity(BaseModel):
    """Allocated generation run and its inputs, independent of student rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: UUID
    generation: GenerationSettings
    agent: AgentSettings
    measurement: RolloutMeasurementSettings
    seed: int
    selections: dict[TaskRole, TaskSelection]
    selection_inputs: dict[TaskRole, TaskSelectionInput]
    tasks: dict[TaskRole, list[Task]]
    expected_pool: PoolManifest
    system_prompt: str
    agent_interface_version: int = AGENT_INTERFACE_VERSION
    plan_fingerprint_version: int = PLAN_FINGERPRINT_VERSION


class PlanOnlyEvidence(BaseModel):
    """Shared validation records and selection, with no invented timing outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    default: Baseline | None
    candidates: list[Candidate]
    kept_default: bool
    selection: SelectionState


class GenerationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attempt_id: str
    split: TaskRole
    task_id: str
    generation_index: int
    model_seed: int
    measurement_seed: int
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None = None
    worker: WorkerManifest | None = None
    trace: AgentTrace | None = None
    rollout: RolloutRecord | None = None
    plan_only: PlanOnlyEvidence | None = None
    error_type: str | None = None
    error: str | None = None


class GenerationSplitReport(BaseModel):
    requested_attempts: int
    completed_attempts: int
    failed_attempts: int
    saved_conversations: int
    conversion_exclusions: dict[str, str]


class GenerationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    training: GenerationSplitReport
    validation: GenerationSplitReport
    worker_pool: PoolManifest | None
    files: dict[str, str]
