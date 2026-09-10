"""Method-specific configuration composed from the settings each subsystem owns."""

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qorl.agent.schemas import AgentSettings
from qorl.evaluation.schemas import EvaluationSettings
from qorl.measure.schemas import CalibrationSettings, RolloutMeasurementSettings
from qorl.model.schemas import (
    AstraInferenceSettings,
    InferenceSettings,
    LocalInferenceSettings,
    ModelProvider,
    ModelSettings,
    OpenRouterInferenceSettings,
)
from qorl.rl.schemas import RlSettings, RlTrainingSettings
from qorl.sft.schemas import (
    GenerationSettings,
    ImportedGenerationSeeds,
    SftTrainingSettings,
)
from qorl.taskset.schemas import TaskRole, TaskSelection, TaskSelectionInput

DEFAULT_EXPERIMENT_SEED = 42
PLACEHOLDER = "FILL_ME_IN"


class ExperimentMethod(StrEnum):
    SFT = "sft"
    RL = "rl"
    EVAL = "eval"
    CALIBRATE = "calibrate"


class RunStage(StrEnum):
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"
    CALIBRATE = "calibrate"


@dataclass(frozen=True)
class RunRequest:
    """An explicitly selected stage and, when supplied, an existing run."""

    stage: RunStage
    number: int | None = None
    checkpoint: Path | None = None
    split: TaskRole | None = None
    resume: bool = False
    resume_from: Path | None = None


@dataclass(frozen=True)
class CreateRequest:
    """Explicit command inputs; creation resolves them into a saved config."""

    name: str
    method: ExperimentMethod
    postgres_config: Path
    pool_config: Path
    tasksets: tuple[str, ...] = ()
    seed: int = DEFAULT_EXPERIMENT_SEED
    base_model_name_or_path: str | None = None
    base_model_revision: str | None = None
    model_provider: ModelProvider = ModelProvider.LOCAL
    adapter_path: Path | None = None
    dataset_from: Path | None = None
    exclude_tasks_from: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ResolvedSelections:
    """Saved IDs and their creation metadata, including imported generation seeds."""

    selections: dict[TaskRole, TaskSelection]
    inputs: dict[TaskRole, TaskSelectionInput]
    imported_generation_seeds: ImportedGenerationSeeds | None
    exclusions: tuple[TaskSelection, ...] = ()


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    method: ExperimentMethod
    seed: int


class ConfigReference(BaseModel):
    """A PostgreSQL/pool config stays authoritative at this referenced location."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path


class EvaluationData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test: TaskSelectionInput


class TrainingData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training: TaskSelectionInput
    validation: TaskSelectionInput
    test: TaskSelectionInput | None = None


class SftData(TrainingData):
    generation: GenerationSettings | None = None
    dataset_from: Path | None = None
    imported_generation_seeds: ImportedGenerationSeeds | None = None

    @model_validator(mode="after")
    def one_dataset_input(self) -> Self:
        """Choose generation or reuse, retaining original generation seeds for reuse."""
        if self.dataset_from is None:
            if self.generation is None or self.imported_generation_seeds is not None:
                raise ValueError(
                    "SFT generation requires data.generation, without imported seeds"
                )
        elif self.generation is not None or self.imported_generation_seeds is None:
            raise ValueError(
                "SFT reuse requires imported_generation_seeds, without generation"
            )
        return self


class ResourceSettings(BaseModel):
    """GPU IDs for training and local serving, not database-worker resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_gpu_ids: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, min_length=1
    )
    serving_gpu_ids: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def distinct_ids(self) -> Self:
        """Do not assign a GPU twice within either process group."""
        for ids in (self.training_gpu_ids, self.serving_gpu_ids):
            if ids is not None and len(ids) != len(set(ids)):
                raise ValueError("resource GPU IDs must not repeat within a group")
        return self


class BaseExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: ClassVar[ExperimentMethod]

    experiment: ExperimentSettings
    postgres: ConfigReference
    pool: ConfigReference

    @model_validator(mode="after")
    def matching_method(self) -> Self:
        """Require the header to match the method-specific configuration model."""
        if self.experiment.method != self.method:
            raise ValueError(f"expected experiment.method={self.method.value}")
        return self


class ModelExperimentConfig(BaseExperimentConfig):
    model: ModelSettings
    agent: AgentSettings
    evaluation: EvaluationSettings
    measurement: RolloutMeasurementSettings
    inference: InferenceSettings
    resources: ResourceSettings | None = None

    @model_validator(mode="after")
    def applicable_model_settings(self) -> Self:
        """Require the selected provider's inference and access settings."""
        if self.model.provider == ModelProvider.LOCAL:
            if (
                not isinstance(self.inference, LocalInferenceSettings)
                or self.resources is None
            ):
                raise ValueError(
                    "local models require local inference, serving, and resources"
                )
            if (
                self.model.base_url is None
                or self.model.request_timeout_seconds is None
            ):
                raise ValueError(
                    "local models require model.base_url and model.request_timeout_seconds"
                )
            if self.resources.serving_gpu_ids is None:
                raise ValueError("local serving requires resources.serving_gpu_ids")
        elif self.model.provider == ModelProvider.OPENROUTER:
            if self.resources is not None or not isinstance(
                self.inference, OpenRouterInferenceSettings
            ):
                raise ValueError(
                    "OpenRouter evaluation requires OpenRouter inference without local resources"
                )
            self.inference.validate_model(self.model)
        else:
            if self.resources is not None or not isinstance(
                self.inference, AstraInferenceSettings
            ):
                raise ValueError(
                    "hosted evaluation requires Astra inference without local serving or GPUs"
                )
            if self.model.name_or_path != "gpt-6-astra":
                raise ValueError("hosted evaluation supports only gpt-6-astra")
            if (
                self.model.base_url is None
                or self.model.request_timeout_seconds is None
                or self.model.api_key_env is None
            ):
                raise ValueError("hosted evaluation requires model API access settings")
            if self.model.revision is not None or self.model.adapter_path is not None:
                raise ValueError("hosted models do not accept revisions or adapters")
        if self.inference.max_tokens > self.model.context_length:
            raise ValueError("inference.max_tokens exceeds model.context_length")
        if self.method != ExperimentMethod.EVAL:
            if (
                self.model.provider != ModelProvider.LOCAL
                or self.model.adapter_path is not None
            ):
                raise ValueError(
                    "training requires a complete local/Hugging Face base model"
                )
            if self.resources is None or self.resources.training_gpu_ids is None:
                raise ValueError("training requires resources.training_gpu_ids")
        elif self.resources is not None and self.resources.training_gpu_ids is not None:
            raise ValueError("standalone evaluation does not train")
        return self


class SftExperimentConfig(ModelExperimentConfig):
    method = ExperimentMethod.SFT

    data: SftData
    training: SftTrainingSettings

    @model_validator(mode="after")
    def distributed_microbatches(self) -> Self:
        """Each training GPU receives a whole number of microbatches."""
        if self.resources is not None and self.resources.training_gpu_ids is not None:
            training_gpu_count = len(self.resources.training_gpu_ids)
            if self.training.batch_size % (
                self.training.micro_batch_size * training_gpu_count
            ):
                raise ValueError(
                    "training.batch_size must be divisible by micro_batch_size "
                    "times the training GPU count"
                )
        return self


class RlExperimentConfig(ModelExperimentConfig):
    method = ExperimentMethod.RL

    data: TrainingData
    training: RlTrainingSettings
    rl: RlSettings

    @model_validator(mode="after")
    def separate_training_and_serving(self) -> Self:
        """RL training and inference use disjoint GPU groups concurrently."""
        if self.resources is not None and (
            set(self.resources.training_gpu_ids or ())
            & set(self.resources.serving_gpu_ids or ())
        ):
            raise ValueError("RL training and serving GPU IDs must be disjoint")
        return self


class EvaluationExperimentConfig(ModelExperimentConfig):
    method = ExperimentMethod.EVAL

    data: EvaluationData


class CalibrationExperimentConfig(BaseExperimentConfig):
    method = ExperimentMethod.CALIBRATE

    data: EvaluationData
    measurement: CalibrationSettings


type ExperimentConfig = (
    SftExperimentConfig
    | RlExperimentConfig
    | EvaluationExperimentConfig
    | CalibrationExperimentConfig
)


@dataclass(frozen=True)
class RunInputs:
    """Validated experiment-owned configuration and resolved task selections."""

    config: ExperimentConfig
    selections: dict[TaskRole, TaskSelection]


CONFIG_MODELS: dict[ExperimentMethod, type[ExperimentConfig]] = {
    ExperimentMethod.SFT: SftExperimentConfig,
    ExperimentMethod.RL: RlExperimentConfig,
    ExperimentMethod.EVAL: EvaluationExperimentConfig,
    ExperimentMethod.CALIBRATE: CalibrationExperimentConfig,
}


def load_config(path: Path) -> ExperimentConfig:
    """Parse TOML once at the file boundary using the declared method's schema."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    header = ExperimentSettings.model_validate(document.get("experiment"))
    return CONFIG_MODELS[header.method].model_validate(document)
