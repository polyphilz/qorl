from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from qorl.postgres.schemas import ExplainResult, PostgresConfigManifest
from qorl.taskset.schemas import BenchmarkId
from qorl.worker_pool.schemas import PoolManifest, WorkerManifest, WorkerPoolConfig

MIN_CALIBRATION_RUNS = 2
ROLLOUT_SCHEMA_VERSION = 2


class DockerHealth(BaseModel):
    """Only the health status is captured; Docker health logs remain private."""

    model_config = ConfigDict(
        extra="ignore", frozen=True, strict=True, hide_input_in_errors=True
    )

    status: str | None = Field(default=None, alias="Status")


type CapturePhase = Literal["pre", "post"]
type RawCommandDocument = JsonValue


class CaptureRecord(BaseModel):
    """Validated evidence; external fields outside the declared projection are dropped."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        strict=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )


class ContainerState(CaptureRecord):
    status: str = Field(validation_alias="Status")
    started_at: str = Field(validation_alias="StartedAt")
    health: str | None = Field(
        default=None, validation_alias=AliasPath("Health", "Status")
    )

    @model_validator(mode="before")
    @classmethod
    def validate_health(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, dict) and "Health" in value:
            DockerHealth.model_validate(value["Health"] or {})
        return value


class ContainerLimits(CaptureRecord):
    cpuset_cpus: str = Field(validation_alias="CpusetCpus")
    cpuset_mems: str = Field(validation_alias="CpusetMems")
    cpu_period: int = Field(validation_alias="CpuPeriod")
    cpu_quota: int = Field(validation_alias="CpuQuota")
    nano_cpus: int = Field(validation_alias="NanoCpus")
    memory: int = Field(validation_alias="Memory")
    memory_swap: int = Field(validation_alias="MemorySwap")
    shm_size: int = Field(validation_alias="ShmSize")
    pids_limit: int | None = Field(validation_alias="PidsLimit")
    readonly_rootfs: bool = Field(validation_alias="ReadonlyRootfs")
    cgroupns_mode: str = Field(validation_alias="CgroupnsMode")


class ContainerMount(CaptureRecord):
    type: str = Field(validation_alias="Type")
    name: str | None = Field(default=None, validation_alias="Name")
    destination: str = Field(validation_alias="Destination")
    driver: str | None = Field(default=None, validation_alias="Driver")
    mode: str | None = Field(default=None, validation_alias="Mode")
    rw: bool = Field(validation_alias="RW")
    propagation: str | None = Field(default=None, validation_alias="Propagation")


class ContainerCapture(CaptureRecord):
    id: str = Field(validation_alias="Id")
    name: str = Field(validation_alias="Name")
    created: str = Field(validation_alias="Created")
    image_id: str = Field(validation_alias="Image")
    image_reference: str = Field(validation_alias=AliasPath("Config", "Image"))
    labels: dict[str, str] = Field(
        default_factory=dict[str, str], validation_alias=AliasPath("Config", "Labels")
    )
    state: ContainerState = Field(validation_alias="State")
    limits: ContainerLimits = Field(validation_alias="HostConfig")
    mounts: list[ContainerMount] = Field(validation_alias="Mounts")

    @field_validator("name")
    @classmethod
    def bare_name(cls, value: str) -> str:
        return value.lstrip("/")

    @field_validator("labels", mode="before")
    @classmethod
    def empty_labels(cls, value: JsonValue) -> JsonValue:
        return {} if value is None else value


class ImageCapture(CaptureRecord):
    id: str = Field(validation_alias="Id")
    repo_tags: list[str] = Field(default_factory=list[str], validation_alias="RepoTags")
    repo_digests: list[str] = Field(
        default_factory=list[str], validation_alias="RepoDigests"
    )
    created: str = Field(validation_alias="Created")
    os: str = Field(validation_alias="Os")
    architecture: str = Field(validation_alias="Architecture")
    size: int = Field(validation_alias="Size")
    labels: dict[str, str] = Field(
        default_factory=dict[str, str], validation_alias=AliasPath("Config", "Labels")
    )
    rootfs_layers: list[str] = Field(
        default_factory=list[str], validation_alias=AliasPath("RootFS", "Layers")
    )

    @field_validator("repo_tags", "repo_digests", "rootfs_layers", mode="before")
    @classmethod
    def empty_lists(cls, value: JsonValue) -> JsonValue:
        return [] if value is None else value

    @field_validator("labels", mode="before")
    @classmethod
    def empty_labels(cls, value: JsonValue) -> JsonValue:
        return {} if value is None else value


class DockerInfo(CaptureRecord):
    server_version: str | None = Field(default=None, alias="ServerVersion")
    driver: str | None = Field(default=None, alias="Driver")
    logging_driver: str | None = Field(default=None, alias="LoggingDriver")
    cgroup_driver: str | None = Field(default=None, alias="CgroupDriver")
    cgroup_version: str | None = Field(default=None, alias="CgroupVersion")
    kernel_version: str | None = Field(default=None, alias="KernelVersion")
    operating_system: str | None = Field(default=None, alias="OperatingSystem")
    os_type: str | None = Field(default=None, alias="OSType")
    architecture: str | None = Field(default=None, alias="Architecture")
    cpu_count: int | None = Field(default=None, alias="NCPU")
    memory_total: int | None = Field(default=None, alias="MemTotal")
    docker_root_dir: str | None = Field(default=None, alias="DockerRootDir")
    name: str | None = Field(default=None, alias="Name")
    live_restore_enabled: bool | None = Field(default=None, alias="LiveRestoreEnabled")
    security_options: list[str] | None = Field(default=None, alias="SecurityOptions")


class ContainerRuntime(CaptureRecord):
    cpus_allowed_list: str
    mems_allowed_list: str
    cpu_max: str
    memory_max: str
    memory_swap_max: str
    shm_size_bytes: str


class CpuPowerPolicy(CaptureRecord):
    policy: str
    affected_cpus: str | None
    scaling_driver: str | None
    scaling_governor: str | None
    energy_performance_preference: str | None
    cpuinfo_min_freq: str | None
    cpuinfo_max_freq: str | None


class GpuCapture(CaptureRecord):
    index: str
    name: str
    uuid: str
    pci_bus_id: str
    driver_version: str
    vbios_version: str


class GpuState(CaptureRecord):
    gpus: list[GpuCapture]
    topology: str


class HostKernel(CaptureRecord):
    system: str
    release: str
    version: str
    machine: str
    command_line: str | None


class HostMachine(CaptureRecord):
    sys_vendor: str | None
    product_name: str | None
    product_version: str | None
    bios_version: str | None


class HostCpu(CaptureRecord):
    summary: RawCommandDocument
    topology: RawCommandDocument
    microcode: str | None
    isolated: str | None
    power_policies: list[CpuPowerPolicy]


class HostMemory(CaptureRecord):
    meminfo: dict[str, str]
    transparent_huge_pages: str | None


class HostStorage(CaptureRecord):
    docker_root_mount: RawCommandDocument
    block_devices: RawCommandDocument
    schedulers: dict[str, str]


class HostCapture(CaptureRecord):
    hostname: str
    os_release: dict[str, str]
    kernel: HostKernel
    machine: HostMachine
    cpu: HostCpu
    memory: HostMemory
    storage: HostStorage
    gpus: GpuState | None


class RuntimeIdentity(CaptureRecord):
    postgres_image_id: str
    postgres_config_id: str


class RuntimeProfile(CaptureRecord):
    id: str
    path: str
    sha256: str
    configuration: WorkerPoolConfig


class DockerCapture(CaptureRecord):
    version: RawCommandDocument
    compose_version: str
    info: DockerInfo
    container: ContainerCapture
    container_runtime: ContainerRuntime
    image: ImageCapture
    image_file_sha256: dict[str, str]


class CaptureArtifact(CaptureRecord):
    sha256: str
    bytes: int


class EnvironmentCapture(CaptureRecord):
    schema_version: Literal[1] = 1
    runtime_identity: RuntimeIdentity
    postgres_config: PostgresConfigManifest
    phase: CapturePhase
    captured_at_utc: str
    assertion_output: str
    runtime_profile: RuntimeProfile
    host: HostCapture
    docker: DockerCapture
    artifacts: dict[str, CaptureArtifact]


class RolloutMeasurementSettings(BaseModel):
    """Initial, optional feedback, and final paired timings with statement timeouts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    default_warmups: int = Field(ge=0)
    default_measurements: int = Field(ge=1)
    candidate_feedback_warmups: int = Field(default=0, ge=0)
    candidate_feedback_measurements: int = Field(default=0, ge=0)
    paired_warmups: int = Field(ge=0)
    paired_measurements: int = Field(ge=1)
    default_timeout_seconds: float = Field(gt=0)
    candidate_timeout_floor_seconds: float = Field(gt=0)
    candidate_timeout_multiplier: float = Field(gt=0)

    @model_validator(mode="after")
    def coherent_feedback(self) -> Self:
        if self.candidate_feedback_warmups and not self.candidate_feedback_measurements:
            raise ValueError(
                "candidate feedback warmups require feedback measurements; use 0+0 to disable"
            )
        return self


class CalibrationSettings(BaseModel):
    """Adaptive warmups followed by repeated default-query measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_warmup_runs: int = Field(ge=MIN_CALIBRATION_RUNS)
    num_trials: int = Field(ge=MIN_CALIBRATION_RUNS)
    default_timeout_seconds: float = Field(gt=0)


class RunStatus(StrEnum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    PASSED = "passed"


class MeasuredPlan(StrEnum):
    DEFAULT = "default"
    CANDIDATE = "candidate"


class MeasurementStatus(StrEnum):
    NOT_MEASURED = "not_measured"
    MEASURED = "measured"


class ToolResultStatus(StrEnum):
    FINISHED = "finished"
    KEPT_DEFAULT = "kept_default"


class OutcomeKind(StrEnum):
    KEPT_DEFAULT = "kept_default"
    DEFAULT_DUPLICATE = "default_duplicate"
    MEASURED = "measured"
    TIMED_OUT = "timed_out"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    SELECTION_FAILED = "selection_failed"


class Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    def to_wire(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


class Measurement(Record):
    execution_time_ms: float = Field(ge=0)
    planning_time_ms: float = Field(ge=0)
    plan_sha256: str
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0
    analyzed_document: dict[str, JsonValue] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    hint_diagnostics: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ExecutionFeedback(Record):
    """Probe evidence; a source reference means these lists contain no new work."""

    status: Literal["collecting", "completed", "timed_out"] = "collecting"
    source_id: str | None = None
    warmups: list[Measurement] = Field(default_factory=list[Measurement])
    measurements: list[Measurement] = Field(default_factory=list[Measurement])
    timeout_ms: int | None = None

    @model_validator(mode="after")
    def coherent_evidence(self) -> Self:
        if self.source_id is not None and (self.warmups or self.measurements):
            raise ValueError(
                "reused feedback references its source instead of copying executions"
            )
        if self.status == "timed_out" and self.timeout_ms is None:
            raise ValueError("timed-out feedback requires its cutoff")
        if (
            self.status == "completed"
            and self.source_id is None
            and not self.measurements
        ):
            raise ValueError("completed feedback requires measurements or a source")
        return self


class ExecutionCounts(BaseModel):
    """Attempted timed SQL statements, including timeouts and infrastructure failures."""

    initial_default: int = Field(default=0, ge=0)
    candidate_feedback: int = Field(default=0, ge=0)
    final_paired: int = Field(default=0, ge=0)


class QueryObservation(Measurement):
    run: int


class CalibrationSummary(BaseModel):
    """Timing variation and plan identity across measured trials, excluding warmups."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    measurement_count: int
    warmup_stable: bool
    median_execution_time_ms: float
    mean_execution_time_ms: float
    sample_standard_deviation_ms: float
    coefficient_of_variation: float | None
    minimum_execution_time_ms: float
    maximum_execution_time_ms: float
    distinct_plan_count: int
    plan_sha256s: list[str]


class CalibrationTaskRecord(BaseModel):
    """One task's actual executions, including partial evidence on failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[2] = 2
    task_id: str
    template_id: str
    completed_at_utc: str
    statement_timeout_ms: int
    worker: WorkerManifest
    warmups: list[QueryObservation]
    measurements: list[QueryObservation]


class CalibratedTask(CalibrationTaskRecord):
    status: Literal[RunStatus.COMPLETED] = RunStatus.COMPLETED
    summary: CalibrationSummary
    representative_explain_analyze: dict[str, JsonValue]


class FailedCalibrationTask(CalibrationTaskRecord):
    status: Literal[RunStatus.FAILED] = RunStatus.FAILED
    error_type: str
    error: str


type CalibrationTaskResult = Annotated[
    CalibratedTask | FailedCalibrationTask, Field(discriminator="status")
]


class CalibrationReport(BaseModel):
    """Progress and applied configuration for one calibration stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal[3] = 3
    benchmark_id: BenchmarkId
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str | None = None
    measurement: CalibrationSettings
    statement_timeout_ms: int
    minimum_warmup_runs: int
    buffer_stability_relative_tolerance: float
    plan_fingerprint_version: int
    worker_pool: PoolManifest | None = None
    task_count: int
    completed_task_count: int = 0
    failed_task_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class QueryRun:
    is_warmup: bool
    observation: QueryObservation
    explain: ExplainResult


class Baseline(Record):
    plan_sha256: str
    structural_plan_sha256: str | None = None
    timing_reuse_key: str | None = None
    plan_fingerprint_version: int | None = None
    plain_explain: dict[str, JsonValue]
    median_execution_time_ms: float | None
    warmups: list[Measurement] = []
    measurements: list[Measurement] = []
    candidate_timeout_ms: int | None = Field(default=None, gt=0)
    compact_plan: dict[str, JsonValue]


class Candidate(Record):
    """One attempt's evidence; duplicate_of denotes execution reuse, not novelty.

    structural_duplicate_of identifies a matching physical plan. plan_sha256
    retains estimates; timing_reuse_key also binds the per-query overrides.
    """

    candidate_id: str
    action: JsonValue
    # Original tool envelope for submissions rejected before PlanAction validation.
    rejected_tool_arguments: JsonValue = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    action_valid: bool
    constraints_satisfied: bool
    compiled_hint: str
    duplicate_of: str | None
    plan_sha256: str | None
    structural_plan_sha256: str | None = None
    structural_duplicate_of: str | None = None
    timing_reuse_key: str | None = None
    plan_fingerprint_version: int | None = None
    errors_or_diagnostics: list[str] = []
    pg_hint_plan: dict[str, str] | None
    attempts_remaining: int
    plain_explain: dict[str, JsonValue] | None = None
    compact_plan: dict[str, JsonValue] | None = None
    execution_timed_out: bool = False

    timeout_ms: int | None = None
    measurement_status: MeasurementStatus | None = None
    execution_feedback: ExecutionFeedback | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def selection_eligible(self) -> bool:
        return self.action_valid and (
            self.constraints_satisfied or self.execution_timed_out
        )

    def feedback(self) -> dict[str, JsonValue]:
        """Expose actionable validation results, not stored hashes or invented timing."""
        return CandidateFeedback(
            candidate_id=self.candidate_id,
            action_valid=self.action_valid,
            constraints_satisfied=self.constraints_satisfied,
            compiled_hint=self.compiled_hint,
            duplicate_of=self.duplicate_of,
            structurally_novel=self.structurally_novel,
            structural_duplicate_of=self.structural_duplicate_of,
            compact_plan=self.compact_plan,
            execution_timed_out=self.execution_timed_out,
            timeout_ms=self.timeout_ms,
            errors_or_diagnostics=self.errors_or_diagnostics,
            attempts_remaining=self.attempts_remaining,
        ).to_wire()

    @property
    def structurally_novel(self) -> bool:
        """Novelty requires validated structure, not merely a different full hash."""
        return (
            self.action_valid
            and self.constraints_satisfied
            and self.structural_plan_sha256 is not None
            and self.structural_duplicate_of is None
        )


class CandidateFeedback(Record):
    candidate_id: str
    action_valid: bool
    constraints_satisfied: bool
    compiled_hint: str
    duplicate_of: str | None
    structurally_novel: bool
    structural_duplicate_of: str | None
    compact_plan: dict[str, JsonValue] | None
    execution_timed_out: bool
    timeout_ms: int | None
    errors_or_diagnostics: list[str]
    attempts_remaining: int


class MeasurementPair(Record):
    """Execution order and completed observations; an interrupted pair may be partial."""

    order: tuple[MeasuredPlan, MeasuredPlan]
    default: Measurement | None = None
    candidate: Measurement | None = None

    @model_validator(mode="after")
    def one_of_each(self) -> Self:
        if set(self.order) != set(MeasuredPlan):
            raise ValueError("pair order must contain default and candidate once each")
        first, second = self.order
        if getattr(self, second) is not None and getattr(self, first) is None:
            raise ValueError("pair observations must follow the recorded order")
        return self


class PairedMeasurements(Record):
    """Final measurement evidence, excluding initial default executions."""

    warmups: list[MeasurementPair] = []
    measurements: list[MeasurementPair] = []


class KeptDefaultOutcome(Record):
    kind: Literal[OutcomeKind.KEPT_DEFAULT] = OutcomeKind.KEPT_DEFAULT
    selected_candidate_id: None = None
    selected_plan_sha256: str
    timing_reuse_key: str
    baseline_reference: Literal["default"] = "default"
    speedup: float = Field(default=1.0, ge=1.0, le=1.0)


class DefaultDuplicateOutcome(Record):
    kind: Literal[OutcomeKind.DEFAULT_DUPLICATE] = OutcomeKind.DEFAULT_DUPLICATE
    selected_candidate_id: str
    selected_plan_sha256: str
    timing_reuse_key: str
    baseline_reference: Literal["default"] = "default"
    speedup: float = Field(default=1.0, ge=1.0, le=1.0)


class MeasuredOutcome(Record):
    kind: Literal[OutcomeKind.MEASURED] = OutcomeKind.MEASURED
    selected_candidate_id: str
    selected_plan_sha256: str
    timing_reuse_key: str
    paired: PairedMeasurements
    default_median_execution_time_ms: float = Field(gt=0)
    candidate_median_execution_time_ms: float = Field(gt=0)
    speedup: float = Field(gt=0)

    @model_validator(mode="after")
    def completed_measurements(self) -> Self:
        if not self.paired.measurements:
            raise ValueError("measured outcome requires paired observations")
        for pair in [*self.paired.warmups, *self.paired.measurements]:
            if pair.default is None or pair.candidate is None:
                raise ValueError("measured outcome requires complete pairs")
        default = statistics.median(
            pair.default.execution_time_ms
            for pair in self.paired.measurements
            if pair.default is not None
        )
        candidate = statistics.median(
            pair.candidate.execution_time_ms
            for pair in self.paired.measurements
            if pair.candidate is not None
        )
        if (
            default != self.default_median_execution_time_ms
            or candidate != self.candidate_median_execution_time_ms
        ):
            raise ValueError("outcome medians must match the measured pairs")
        if not math.isclose(self.speedup, default / candidate):
            raise ValueError(
                "speedup must equal the unclipped default/candidate median ratio"
            )
        return self


class TimedOutOutcome(Record):
    kind: Literal[OutcomeKind.TIMED_OUT] = OutcomeKind.TIMED_OUT
    selected_candidate_id: str
    selected_plan_sha256: str | None
    timing_reuse_key: str | None
    speedup: None = None
    initial_default_median_execution_time_ms: float = Field(gt=0)
    timeout_ms: int = Field(gt=0)
    paired: PairedMeasurements


class NoValidCandidateOutcome(Record):
    kind: Literal[OutcomeKind.NO_VALID_CANDIDATE] = OutcomeKind.NO_VALID_CANDIDATE
    selected_candidate_id: None = None
    selected_plan_sha256: None = None
    timing_reuse_key: None = None
    speedup: None = None


class RejectedSelection(Record):
    arguments: JsonValue
    diagnostics: list[str]


class SelectionStatus(StrEnum):
    PENDING = "pending"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    FAILED = "failed"


class SelectionState(BaseModel):
    status: SelectionStatus = SelectionStatus.PENDING
    selected_candidate_id: str | None = None
    rejections: list[RejectedSelection] = Field(default_factory=list[RejectedSelection])


class SelectionFailedOutcome(Record):
    kind: Literal[OutcomeKind.SELECTION_FAILED] = OutcomeKind.SELECTION_FAILED
    selected_candidate_id: None = None
    selected_plan_sha256: None = None
    timing_reuse_key: None = None
    speedup: None = None


type Outcome = Annotated[
    KeptDefaultOutcome
    | DefaultDuplicateOutcome
    | MeasuredOutcome
    | TimedOutOutcome
    | NoValidCandidateOutcome
    | SelectionFailedOutcome,
    Field(discriminator="kind"),
]


class RolloutFailure(Record):
    """Unscored failure with partial evidence, never a policy outcome."""

    operation: str
    error_type: str
    error: str
    paired: PairedMeasurements


class RolloutRecord(Record):
    """One task, its attempt history and exactly one final outcome or unscored failure."""

    schema_version: Literal[2] = ROLLOUT_SCHEMA_VERSION
    task_id: str
    template_id: str
    measurement: RolloutMeasurementSettings
    default: Baseline | None
    candidates: list[Candidate]
    final: Outcome | None
    selection: SelectionState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    failure: RolloutFailure | None = None
    execution_counts: ExecutionCounts | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def evidence_matches_outcome(self) -> Self:
        if (self.final is None) == (self.failure is None):
            raise ValueError("rollout requires one final outcome or unscored failure")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        final = self.final
        if final is None:
            return self
        baseline = self.default
        if baseline is None or baseline.median_execution_time_ms is None:
            raise ValueError("policy outcome requires a measured initial default")
        if (
            len(baseline.warmups) != self.measurement.default_warmups
            or len(baseline.measurements) != self.measurement.default_measurements
        ):
            raise ValueError(
                "initial default observations must match measurement settings"
            )
        if (
            statistics.median(item.execution_time_ms for item in baseline.measurements)
            != baseline.median_execution_time_ms
        ):
            raise ValueError("initial default median must match its observations")
        selected = next(
            (
                item
                for item in self.candidates
                if item.candidate_id == final.selected_candidate_id
            ),
            None,
        )
        if final.selected_candidate_id is not None:
            if selected is None:
                raise ValueError(
                    "selected_candidate_id must reference a recorded attempt"
                )
            if (selected.plan_sha256, selected.timing_reuse_key) != (
                final.selected_plan_sha256,
                final.timing_reuse_key,
            ):
                raise ValueError(
                    "selected plan identity must match its recorded attempt"
                )
        if isinstance(final, (MeasuredOutcome, DefaultDuplicateOutcome)) and (
            selected is None
            or not selected.action_valid
            or not selected.constraints_satisfied
            or selected.execution_timed_out
        ):
            raise ValueError("completed candidate outcome requires a usable candidate")
        if isinstance(final, (KeptDefaultOutcome, DefaultDuplicateOutcome)) and (
            final.selected_plan_sha256,
            final.timing_reuse_key,
        ) != (
            baseline.plan_sha256,
            baseline.timing_reuse_key,
        ):
            raise ValueError(
                "baseline reuse requires matching plan and timing-reuse key"
            )
        if isinstance(final, NoValidCandidateOutcome) and any(
            item.selection_eligible for item in self.candidates
        ):
            raise ValueError("no_valid_candidate cannot discard a usable attempt")
        if isinstance(final, SelectionFailedOutcome) and (
            self.selection is None or self.selection.status != "failed"
        ):
            raise ValueError(
                "selection_failed requires unsuccessful selection evidence"
            )
        if isinstance(final, MeasuredOutcome):
            if final.timing_reuse_key == baseline.timing_reuse_key:
                raise ValueError("measured candidate must require its own timing")
            if (
                len(final.paired.warmups) != self.measurement.paired_warmups
                or len(final.paired.measurements)
                != self.measurement.paired_measurements
            ):
                raise ValueError("completed pairs must match measurement settings")
            for pair in [*final.paired.warmups, *final.paired.measurements]:
                if (
                    pair.default is None
                    or pair.candidate is None
                    or pair.default.plan_sha256 != baseline.plan_sha256
                    or pair.candidate.plan_sha256 != final.selected_plan_sha256
                ):
                    raise ValueError(
                        "paired observations must match the selected plans"
                    )
        if isinstance(final, TimedOutOutcome):
            if (
                selected is None
                or not selected.execution_timed_out
                or selected.timeout_ms != final.timeout_ms
            ):
                raise ValueError("timeout must match the selected attempt's cutoff")
            if (
                final.initial_default_median_execution_time_ms
                != baseline.median_execution_time_ms
            ):
                raise ValueError("timeout must reference the initial default median")
        return self
