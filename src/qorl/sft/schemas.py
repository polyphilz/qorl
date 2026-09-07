from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from qorl.measure.schemas import Baseline, Candidate, RunStatus
from qorl.model.schemas import ModelProvider, ModelSettings, TrainerModelSettings
from qorl.taskset.schemas import TaskSelection

type JsonObject = dict[str, JsonValue]

JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
JSON_OBJECT_LIST_ADAPTER: TypeAdapter[list[JsonObject]] = TypeAdapter(list[JsonObject])
STRING_LIST_ADAPTER: TypeAdapter[list[str]] = TypeAdapter(list[str])
GENERATOR_MODELS = {
    ModelProvider.OPENAI: "gpt-6-astra",
    ModelProvider.ANTHROPIC: "claude-fable-5-1",
}


class SftTrainingSettings(BaseModel):
    """Epochs and packed-row batch sizes, not original conversation counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=1)
    model: TrainerModelSettings

    @model_validator(mode="after")
    def whole_microbatches(self) -> Self:
        """An optimizer batch contains whole microbatches of packed rows."""
        if self.batch_size % self.micro_batch_size:
            raise ValueError(
                "training.batch_size must be divisible by micro_batch_size"
            )
        return self


class GenerationSettings(BaseModel):
    """Explicit hosted generator; acceptance policy is gated on plan 032."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelSettings | Literal["FILL_ME_IN"]
    generations_per_task: Annotated[int, Field(ge=1)] | Literal["FILL_ME_IN"]

    @model_validator(mode="after")
    def hosted_generator(self) -> Self:
        """Generation uses a hosted model, never the trainee or a local adapter."""
        if isinstance(self.model, ModelSettings):
            if self.model.provider == ModelProvider.LOCAL:
                raise ValueError("SFT generation requires an OpenAI or Anthropic model")
            if self.model.name_or_path != GENERATOR_MODELS[self.model.provider]:
                raise ValueError(
                    "SFT generation supports only GPT-6 Astra or Claude Fable 5.1"
                )
            if self.model.revision is not None or self.model.adapter_path is not None:
                raise ValueError(
                    "hosted generators do not accept revisions or adapters"
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

    schema_version: Literal[1]
    format: Literal["qorl-conversations"]
    training: PreparedDatasetSplit
    validation: PreparedDatasetSplit
    tools: Path


class ImportedGenerationSeeds(BaseModel):
    """Keep source generation seeds distinct from a new experiment's seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training: int
    validation: int


class SamplingMode(StrEnum):
    NORMAL = "normal"
    DEFAULT_BEST = "default_best"


class CandidateLabel(StrEnum):
    WIN = "win"
    KNOWN_REGRESSION = "known_regression"
    AMBIGUOUS = "ambiguous"


class TaskLabel(StrEnum):
    KNOWN_WIN = "known_win"
    DEFAULT_BEST = "default_best"
    INSUFFICIENT_FINGERPRINTS = "insufficient_fingerprints"
    AMBIGUOUS = "ambiguous"


class ExampleKind(StrEnum):
    SYNTAX = "syntax"
    WIN = "win"
    KEEP_DEFAULT = "keep_default"


class ExampleSource(StrEnum):
    STUDENT = "student"
    TEACHER = "teacher"


class ActionFamily(StrEnum):
    LEADING = "leading"
    JOIN = "join"
    MEMOIZE = "memoize"
    SCAN = "scan"
    INDEX_SELECTION = "index_selection"
    INDEX_EXCLUSION = "index_exclusion"
    ROWS = "rows"
    PARALLEL = "parallel"
    SETTING = "setting"


class TeacherAttemptStatus(StrEnum):
    PROVIDER_ERROR = "provider_error"
    RESPONSE_ERROR = "response_error"
    REPLAY_ERROR = "replay_error"
    VALIDATION_ERROR = "validation_error"
    ACCEPTED = "accepted"


class SftRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_wire(self) -> JsonObject:
        return JSON_OBJECT_ADAPTER.validate_python(self.model_dump(mode="json"))


class FileIdentity(SftRecord):
    path: str
    sha256: str


class PipelineError(SftRecord):
    type: str
    message: str


class SamplerIdentity(SftRecord):
    model: str
    manifest_sha256: str
    server_identity: JsonObject


class SampleRecord(SftRecord):
    schema_version: Literal[2] = 2
    status: RunStatus
    completed_at_utc: str
    task_id: str
    template_id: str
    sample: int = Field(ge=1)
    seed: int
    sampling_mode: SamplingMode
    steered: Literal[False]
    guidance: None
    worker: JsonObject
    data_identity: JsonObject
    runtime_identity: JsonObject
    sampler: SamplerIdentity
    default: Baseline | None
    candidates: list[Candidate]
    policy_trace: JsonObject | None
    training_transcript: list[JsonObject] | None
    error: PipelineError | None


class TeacherTargets(SftRecord):
    leading: int = Field(ge=1)
    join: int = Field(ge=1)
    memoize: int = Field(ge=1)
    parallel: int = Field(ge=1)

    def as_families(self) -> dict[ActionFamily, int]:
        return {
            ActionFamily.LEADING: self.leading,
            ActionFamily.JOIN: self.join,
            ActionFamily.PARALLEL: self.parallel,
        }


class TeacherConfig(SftRecord):
    schema_version: Literal[1] = 1
    teacher_id: str
    base_url: str
    model: str
    decoding: Literal["provider_default"]
    max_tokens: int = Field(ge=1)
    request_timeout_seconds: int = Field(ge=1)
    provider_retry_delay_seconds: int = Field(ge=0)
    maximum_attempts_per_task: int = Field(ge=1)
    attempt_budget_multiplier: int = Field(ge=1)
    smoke_accepted_per_family: int = Field(ge=1)
    maximum_teacher_share: float = Field(gt=0, lt=1)
    accepted_targets: TeacherTargets
    priority_templates: list[str]


class TeacherIdentity(SftRecord):
    teacher_id: str
    model: str
    base_url: str
    decoding: Literal["provider_default"]
    config: FileIdentity


class TeacherPrefix(SftRecord):
    sample_path: str
    sample_sha256: str
    sample: int = Field(ge=1)
    assistant_turns: int = Field(ge=0)


class TeacherAttempt(SftRecord):
    attempt: int = Field(ge=1)
    completed_at_utc: str
    status: TeacherAttemptStatus
    prompt: str
    tool_schema_sha256: str
    response: JsonObject | None
    action: JsonObject | None
    candidate: Candidate | None
    rejection_reason: str | None


class TeacherGenerationRecord(SftRecord):
    schema_version: Literal[1] = 1
    task_id: str
    template_id: str
    requested_family: ActionFamily
    teacher: TeacherIdentity
    prefix: TeacherPrefix
    attempts: list[TeacherAttempt]
    accepted_sample: SampleRecord | None


class TeacherRecordIdentity(SftRecord):
    task_id: str
    template_id: str
    requested_family: ActionFamily
    path: str
    sha256: str
    accepted: bool


class TeacherSummary(SftRecord):
    tasks_attempted: int = Field(ge=0)
    api_attempts: int = Field(ge=0)
    accepted: int = Field(ge=0)
    accepted_by_family: dict[ActionFamily, int]
    rejected_by_reason: dict[str, int]


class TeacherManifest(SftRecord):
    schema_version: Literal[1] = 1
    generation_id: str
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None
    teacher: TeacherIdentity
    source_filter: FileIdentity
    targets: dict[ActionFamily, int]
    database_pool: JsonObject | None
    summary: TeacherSummary
    records: list[TeacherRecordIdentity]


class SamplingSummary(SftRecord):
    rollouts: int
    completed_rollouts: int
    failed_rollouts: int
    intervention_rollouts: int
    keep_default_rollouts: int
    action_valid_candidates: int
    constraint_satisfied_candidates: int
    default_duplicate_candidates: int
    novel_candidates: int
    distinct_novel_fingerprints: int
    distinct_novel_fingerprint_yield: float
    distinct_fingerprints_per_intervened_task: float
    mixed_decision_task_share: float
    leading_attempts: int
    leading_constraint_satisfied_candidates: int
    action_families: dict[ActionFamily, int]


class FallbackCheck(SftRecord):
    threshold: float
    passed: bool
    summary: SamplingSummary


class SamplingInvocation(SftRecord):
    completed_at_utc: str
    task_ids: list[str]
    sample_start: int
    sample_count: int
    sampling_mode: SamplingMode
    sampler: SamplerIdentity
    guidance: None
    summary: SamplingSummary


class SamplingManifest(SftRecord):
    schema_version: Literal[1] = 1
    sampling_id: str
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None
    split: str
    selection: FileIdentity
    dataset_config: FileIdentity
    policy_config: FileIdentity
    sampler: SamplerIdentity
    sampler_manifest_path: str
    sampler_manifest: JsonObject
    guidance: None
    data_identity: JsonObject
    runtime_identity: JsonObject
    plan_fingerprint_version: int
    database_pool: JsonObject | None
    summary: SamplingSummary
    fallback_check: FallbackCheck | None
    invocations: list[SamplingInvocation]


class FilterRecord(SftRecord):
    task_id: str
    template_id: str
    sample: int = Field(ge=1)
    sample_path: str
    accepted: bool
    rejection_reason: str | None
    plan_sha256: str | None
    structural_plan_sha256: str | None = None
    action_families: list[ActionFamily]
    syntax_eligible: bool
    steered: Literal[False]


class FilterSummary(SftRecord):
    rollouts: int
    accepted_distinct_novel_candidates: int
    accepted_yield: float
    tasks_with_accepted_candidate: int
    tasks_reaching_default_best_budget: int
    rejection_reasons: dict[str, int]
    action_families: dict[ActionFamily, int]


class FilterManifest(SftRecord):
    schema_version: Literal[1] = 1
    filter_id: str
    plan_fingerprint_version: int
    split: str
    source: ExampleSource
    source_manifest: FileIdentity | None
    config_sha256: str
    records_sha256: str
    summary: FilterSummary


class SamplingSettings(SftRecord):
    concurrency: int = Field(ge=1)
    initial_samples_per_task: int = Field(ge=1)
    normal_maximum_samples_per_task: int = Field(ge=1)
    default_best_search_maximum_samples_per_task: int = Field(ge=1)
    default_best_search_minimum_measured_fingerprints: int = Field(ge=1)
    default_best_search_maximum_observed_speedup: float = Field(gt=0)
    default_best_search_task_limit: int = Field(ge=1)
    fallback_check_task_count: int = Field(ge=1)
    fallback_yield_floor: float = Field(ge=0, le=1)


class LabelSettings(SftRecord):
    win_speedup: float
    remeasure_lower_speedup: float
    remeasure_upper_speedup: float
    default_best_maximum_speedup: float
    default_best_minimum_fingerprints: int = Field(ge=1)


class AssemblySettings(SftRecord):
    maximum_examples_per_task: int = Field(ge=1)
    maximum_syntax_examples_per_task: int = Field(ge=1)
    keep_default_examples: int = Field(ge=0)


class MeasurementSettings(SftRecord):
    maximum_candidates_per_task: int = Field(ge=1)
    concurrency: int = Field(ge=1)


class CompositionSettings(SftRecord):
    win_share: float = Field(ge=0, le=1)
    keep_default_share: float = Field(ge=0, le=1)


class SplitCounts(SftRecord):
    train: int = Field(ge=1)
    validation: int = Field(ge=1)


class TrainingSettings(SftRecord):
    validation_points: int = Field(ge=1)
    maximum_validation_interval: int = Field(ge=1)


class GateSettings(SftRecord):
    samples_per_task: int = Field(ge=1)
    concurrency: int = Field(ge=1)


class DatasetConfig(SftRecord):
    schema_version: Literal[1] = 1
    dataset_id: str
    seed: int
    policy_config: str
    selection: str
    sampling: SamplingSettings
    labels: LabelSettings
    assembly: AssemblySettings
    measurement: MeasurementSettings
    composition: CompositionSettings
    split_counts: SplitCounts
    training: TrainingSettings
    gate: GateSettings


class SelectionTask(SftRecord):
    task_id: str
    template_id: str


class SelectionSplits(SftRecord):
    sampling: list[SelectionTask]
    live_gate: list[SelectionTask]
    validation: list[SelectionTask]


class SelectionCounts(SftRecord):
    live_gate: int = Field(ge=1)
    live_gate_by_relation_count: dict[str, int]
    sampling: int = Field(ge=1)
    sampling_by_relation_count: dict[str, int]
    templates: int = Field(ge=1)
    validation: int = Field(ge=1)
    validation_by_relation_count: dict[str, int]


class ExcludedTask(SftRecord):
    reason: str
    task_id: str


class SelectionMethod(SftRecord):
    algorithm: str
    excluded_tasks: list[ExcludedTask]
    live_gate_template_quotas: dict[str, int]
    relation_count_note: str
    rl_v3_exclusion_splits: list[str]
    salt: str
    sampling_tasks_per_template: int = Field(ge=1)
    template_order: list[str]


class SelectionSource(SftRecord):
    inventory_id: str
    path: str
    sha256: str


class DatasetSelection(SftRecord):
    schema_version: Literal[1] = 1
    inventory_id: str
    counts: SelectionCounts
    selection: SelectionMethod
    source: SelectionSource
    splits: SelectionSplits


class TimeoutAlgorithm(SftRecord):
    global_cap_ms: int = Field(ge=1)
    minimum_ms: int = Field(ge=1)
    multiplier: float = Field(gt=0)


class TimeoutSelection(SftRecord):
    inventory_id: str
    path: str
    sha256: str
    split: str


class SourceCalibration(SftRecord):
    calibration_id: str
    manifest_sha256: str
    derived_from: list[JsonObject]


class TimeoutTask(SftRecord):
    task_id: str
    template_id: str
    calibrated_default_median_ms: float
    timeout_ms: int = Field(ge=1)
    plan_sha256s: list[str]


class TimeoutManifest(SftRecord):
    schema_version: Literal[1] = 1
    manifest_id: str
    algorithm: TimeoutAlgorithm
    selection: TimeoutSelection
    data_identity: JsonObject
    runtime_identity: JsonObject
    source_calibration: SourceCalibration
    task_count: int = Field(ge=1)
    tasks: list[TimeoutTask]


class FilterProvenance(SftRecord):
    accepted: bool
    rejection_reason: str | None
    syntax_eligible: bool
    action_families: list[ActionFamily]


class DemonstrationProvenance(SftRecord):
    source: ExampleSource
    sample: int = Field(ge=1)
    sampler: SamplerIdentity
    filter: FilterProvenance
    budget: JsonObject


class CandidateEvidence(SftRecord):
    action: JsonObject
    plain_explain: JsonObject
    pg_hint_plan: dict[str, str] | None


class DemonstrationEvidence(SftRecord):
    default_plan: JsonObject
    candidates: dict[str, CandidateEvidence]


class DemonstrationMetadata(SftRecord):
    demonstration_id: str
    ordinal: int = Field(ge=0)
    teacher: str
    task_set_id: str
    task_id: str
    template_id: str
    partition: str
    sql_sha256: str
    data_identity: JsonObject
    runtime_identity: JsonObject
    in_author_unique_plans_subset: bool
    trace_seed: int
    maximum_model_turns: int = Field(ge=1)
    candidate_count: int = Field(ge=0, le=1)
    measurement_mode: str
    selection_used_speed: bool
    example_kind: ExampleKind
    call_sequence: list[str]


class DemonstrationDocument(SftRecord):
    schema_version: Literal[1] = 1
    messages: list[JsonObject]
    tools: list[JsonObject]
    metadata: DemonstrationMetadata
    provenance: DemonstrationProvenance
    evidence: DemonstrationEvidence


class PrimeArtifact(SftRecord):
    path: str
    rows: int = Field(ge=0)
    bytes: int = Field(ge=0)
    sha256: str


class DatasetSelectionIdentity(SftRecord):
    path: str
    sha256: str
    rl_v3_excluded_train_task_count: int = Field(ge=0)
    sampling_live_gate_disjoint: bool


class DatasetInputs(SftRecord):
    sampling_manifest_sha256: str
    sampling_filter_manifest_sha256: str
    teacher_filter_manifest_sha256: str


class DemonstrationIdentity(SftRecord):
    demonstration_id: str
    partition: str
    task_id: str
    template_id: str
    path: str
    canonical_sha256: str


class DatasetManifest(SftRecord):
    schema_version: Literal[1] = 1
    dataset_id: str
    seed: int
    config: FileIdentity
    selection: DatasetSelectionIdentity
    task_set_id: str
    counts: dict[str, int]
    composition: dict[str, dict[ExampleKind, int]]
    templates: dict[str, dict[str, int]]
    train_action_families: dict[ActionFamily, int]
    train_example_sources: dict[ExampleSource, int]
    prime_artifacts: dict[str, PrimeArtifact]
    inputs: DatasetInputs
    demonstrations: list[DemonstrationIdentity]


class PreparationReport(SftRecord):
    optimizer_steps: int = Field(ge=1)
    render_audit: str
    resolved_config: str


class TrainingReport(SftRecord):
    schema_version: Literal[1] = 1
    status: RunStatus
    prime_rl_version: str
    model: str
    model_revision: str
    optimizer_steps: int = Field(ge=1)
    peak_gpu_memory_gib: float
    final_training_loss: float
    dataset_manifest_sha256: str
    render_audit_sha256: str
    resolved_config_sha256: str
    adapter: str
    adapter_verification: str


class GateRollout(SftRecord):
    task_id: str
    template_id: str
    cohort: str
    sample: int = Field(ge=1)
    seed: int
    status: RunStatus
    decision: str | None
    action_valid: bool
    constraints_satisfied: bool
    default_duplicate: bool
    novel_fingerprint: str | None
    action_families: list[ActionFamily]
    error: PipelineError | None


class GateSummary(SftRecord):
    task_count: int = Field(ge=0)
    rollout_count: int = Field(ge=0)
    completed_rollouts: int = Field(ge=0)
    failed_rollouts: int = Field(ge=0)
    valid_plan_rate: float
    novel_plan_rate: float


class GateReport(SftRecord):
    plan_fingerprint_version: int
    schema_version: Literal[1] = 1
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None
    model: str
    server_identity: JsonObject
    selection: FileIdentity
    dataset_config: FileIdentity
    summary: GateSummary | None
    database_pool: JsonObject | None
    rollouts: list[GateRollout]


def load_record[RecordT: BaseModel](path: Path, record: type[RecordT]) -> RecordT:
    return record.model_validate_json(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


def load_string_list(path: Path) -> list[str]:
    return STRING_LIST_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


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


def require_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    return value


def require_int(value: JsonValue, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_float(value: JsonValue, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    return float(value)
