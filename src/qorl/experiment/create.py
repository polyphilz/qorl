"""Create an experiment using only local configuration, catalogs, and artifacts."""

import re
import shutil
import tomllib
from pathlib import Path

import tomli_w

from qorl.adapters.config import adapter_config
from qorl.experiment.schemas import (
    PLACEHOLDER,
    CalibrationExperimentConfig,
    ConfigReference,
    CreateRequest,
    EvaluationData,
    EvaluationExperimentConfig,
    ExperimentConfig,
    ExperimentMethod,
    ExperimentSettings,
    ResolvedSelections,
    RlExperimentConfig,
    SftData,
    SftExperimentConfig,
    TrainingData,
    load_config,
)
from qorl.model.files import validate_model_directory
from qorl.model.schemas import ModelPreset, ModelProvider, ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.config import PostgresConfig
from qorl.sft.schemas import ImportedGenerationSeeds, PreparedDatasetManifest
from qorl.taskset.schemas import (
    BenchmarkId,
    TaskRole,
    TaskSelection,
    TaskSelectionInput,
)
from qorl.taskset.selection import parse_selection, select_tasks, validate_splits
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import load_pool_config

DEFAULTS_DIRECTORY = REPOSITORY_ROOT / "configs/defaults"
EXPERIMENTS_DIRECTORY = REPOSITORY_ROOT / "experiments"
EXPERIMENT_NUMBER_WIDTH = 3
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
TASK_FILES = {
    TaskRole.TRAIN: "training-tasks.json",
    TaskRole.VALIDATION: "validation-tasks.json",
    TaskRole.TEST: "test-tasks.json",
}
RUN_SCRIPT = '''"""Delegate this experiment to QORL's shared stage implementation."""

from pathlib import Path

from qorl.experiment.run import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
'''


def latest_template(method: ExperimentMethod) -> Path:
    """Select the method family's highest numeric version, ignoring mtime."""
    family = "calibration" if method == ExperimentMethod.CALIBRATE else method.value
    return latest_config(DEFAULTS_DIRECTORY, family)


def latest_config(directory: Path, family: str) -> Path:
    """Select one numbered template or model preset by version, never modification time."""
    pattern = re.compile(rf"(?P<version>[0-9]+)-{re.escape(family)}\.toml")
    versions: dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        version = int(match["version"])
        if version in versions:
            raise ValueError(
                f"duplicate numeric template version for {family}: {version}"
            )
        versions[version] = path
    if not versions:
        raise ValueError(f"no default template for {family}")
    return versions[max(versions)]


def recorded_path(path: Path) -> Path:
    """Keep repository-local references relative; retain absolute external paths."""
    path = path.resolve()
    return (
        path.relative_to(REPOSITORY_ROOT)
        if path.is_relative_to(REPOSITORY_ROOT)
        else path
    )


def local_path(path: Path) -> Path:
    """Resolve configured local inputs against the repository, not the caller's cwd."""
    return (REPOSITORY_ROOT / path.expanduser()).resolve()


def artifact_file(directory: Path, relative: Path) -> Path:
    """Require an artifact member to be a file contained in that artifact."""
    path = (directory / relative).resolve()
    if (
        relative.is_absolute()
        or not path.is_relative_to(directory)
        or not path.is_file()
    ):
        raise ValueError(
            f"missing or external artifact file: {relative} in {directory}"
        )
    return path


def model_settings(request: CreateRequest, template: ModelSettings) -> ModelSettings:
    """Resolve a local model or record a pinned remote identity without downloading."""
    name = request.base_model_name_or_path
    if not name or name == PLACEHOLDER:
        raise ValueError(
            "--base-model-name-or-path is required for model-based experiments"
        )
    revision = request.base_model_revision
    if request.model_provider != ModelProvider.LOCAL:
        if request.method != ExperimentMethod.EVAL:
            raise ValueError(
                "hosted base models are only valid for standalone evaluation"
            )
        if revision is not None or request.adapter_path is not None:
            raise ValueError(
                "hosted models do not accept --base-model-revision or --adapter-path"
            )
    else:
        path = local_path(Path(name))
        if path.exists() or name.startswith(("/", ".", "~")):
            if revision is not None:
                raise ValueError(
                    "--base-model-revision applies only to Hugging Face IDs"
                )
            validate_model_directory(path)
            name = str(path)
        elif revision is None or REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError(
                "Hugging Face models require --base-model-revision as a 40-character commit"
            )
    adapter = None
    if request.adapter_path is not None:
        if request.method != ExperimentMethod.EVAL:
            raise ValueError(
                "--adapter-path is only valid for standalone local evaluation"
            )
        adapter = local_path(request.adapter_path)
        adapter_config(adapter)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError(f"adapter weights missing: {adapter}")
    return ModelSettings(
        provider=request.model_provider,
        name_or_path=name,
        revision=revision,
        adapter_path=adapter,
        context_length=template.context_length,
        base_url=template.base_url
        if request.model_provider == template.provider
        else None,
        request_timeout_seconds=template.request_timeout_seconds
        if request.model_provider == template.provider
        else None,
        api_key_env=template.api_key_env
        if request.model_provider == template.provider
        else None,
        max_concurrent_requests=template.max_concurrent_requests,
        retry=template.retry,
    )


def resolve_selections(request: CreateRequest) -> ResolvedSelections:
    """Select new IDs or inherit dataset splits, then enforce method roles and topology."""
    catalogs = {
        benchmark.value: TaskSet.load(REPOSITORY_ROOT, benchmark.value)
        for benchmark in BenchmarkId
    }
    selections: dict[TaskRole, TaskSelection] = {}
    inputs: dict[TaskRole, TaskSelectionInput] = {}
    generation_seeds = None
    if request.dataset_from is not None:
        source = local_path(request.dataset_from)
        manifest = PreparedDatasetManifest.model_validate_json(
            (source / "manifest.json").read_bytes()
        )
        for role, split in (
            (TaskRole.TRAIN, manifest.training),
            (TaskRole.VALIDATION, manifest.validation),
        ):
            artifact_file(source, split.conversations)
            selections[role] = split.selection
            inputs[role] = TaskSelectionInput(
                file=Path(TASK_FILES[role]),
                selection_seed=split.selection_seed,
                expression=split.selection_expression,
            )
        generation_seeds = ImportedGenerationSeeds(
            training=manifest.training.generation_seed,
            validation=manifest.validation.generation_seed,
        )
    for expression in request.tasksets:
        parsed = parse_selection(expression)
        if request.dataset_from is not None and parsed.role != TaskRole.TEST:
            raise ValueError(
                "--dataset-from accepts only an optional test selection; train/validation are inherited"
            )
        inputs[parsed.role] = TaskSelectionInput(
            file=Path(TASK_FILES[parsed.role]),
            selection_seed=request.seed,
            expression=expression,
        )
    if request.tasksets:
        selections.update(select_tasks(request.tasksets, catalogs, request.seed))
    if request.method in (ExperimentMethod.SFT, ExperimentMethod.RL):
        if not {TaskRole.TRAIN, TaskRole.VALIDATION}.issubset(selections):
            raise ValueError(
                "training experiments require train and validation tasksets"
            )
    elif set(selections) != {TaskRole.TEST}:
        raise ValueError("evaluation and calibration require only a test taskset")
    validate_splits(selections, catalogs)
    return ResolvedSelections(selections, inputs, generation_seeds)


def configure(
    request: CreateRequest,
    template: ExperimentConfig,
    selected: ResolvedSelections,
) -> ExperimentConfig:
    """Apply supplied inputs to typed defaults and validate the resulting method config."""
    postgres = PostgresConfig.load(local_path(request.postgres_config))
    pool = load_pool_config(local_path(request.pool_config))
    config = template.model_copy(
        update={
            "experiment": ExperimentSettings(
                name=request.name, method=request.method, seed=request.seed
            ),
            "postgres": ConfigReference(path=recorded_path(postgres.path)),
            "pool": ConfigReference(path=recorded_path(local_path(pool.path))),
        }
    )
    if isinstance(config, (SftExperimentConfig, RlExperimentConfig)):
        training_data = TrainingData(
            training=selected.inputs[TaskRole.TRAIN],
            validation=selected.inputs[TaskRole.VALIDATION],
            test=selected.inputs.get(TaskRole.TEST),
        )
        if isinstance(config, SftExperimentConfig):
            data = SftData(
                training=training_data.training,
                validation=training_data.validation,
                test=training_data.test,
                generation=config.data.generation
                if request.dataset_from is None
                else None,
                dataset_from=recorded_path(local_path(request.dataset_from))
                if request.dataset_from is not None
                else None,
                imported_generation_seeds=selected.imported_generation_seeds,
            )
            config = config.model_copy(update={"data": data})
        else:
            config = config.model_copy(update={"data": training_data})
    else:
        config = config.model_copy(
            update={"data": EvaluationData(test=selected.inputs[TaskRole.TEST])}
        )
    if not isinstance(config, CalibrationExperimentConfig):
        if request.model_provider == ModelProvider.OPENAI:
            if request.method != ExperimentMethod.EVAL:
                raise ValueError(
                    "hosted base models are only valid for standalone evaluation"
                )
            if request.base_model_name_or_path != "gpt-6-astra":
                raise ValueError("hosted evaluation supports only gpt-6-astra")
            preset_path = latest_config(
                DEFAULTS_DIRECTORY / "models", request.base_model_name_or_path
            )
            preset = ModelPreset.model_validate(tomllib.loads(preset_path.read_text()))
            if (
                preset.model.provider != request.model_provider
                or preset.model.name_or_path != request.base_model_name_or_path
            ):
                raise ValueError(
                    "model preset does not match the selected provider/model"
                )
            config = config.model_copy(
                update={
                    "model": preset.model,
                    "inference": preset.inference,
                    "resources": None,
                }
            )
        config = config.model_copy(
            update={"model": model_settings(request, config.model)}
        )
    return type(config).model_validate(config.model_dump())


def readme(directory: Path, template_path: Path, config: ExperimentConfig) -> str:
    """Describe the saved inputs and explicit stage commands, without run claims."""
    gpu = (
        not isinstance(config, CalibrationExperimentConfig)
        and config.model.provider == ModelProvider.LOCAL
    )
    launcher = "uv run --extra gpu" if gpu else "uv run"
    command = f"{launcher} qorl experiment run experiments/{directory.name}"
    if isinstance(config, SftExperimentConfig):
        commands = [
            f"{command} --stage prepare",
            f"{command} --stage train --run 000",
            f"{command} --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split validation",
        ]
    elif isinstance(config, RlExperimentConfig):
        commands = [
            f"{command} --stage train",
            f"{command} --stage evaluate --run 000 --checkpoint /path/to/checkpoint --split validation",
        ]
    elif isinstance(config, EvaluationExperimentConfig):
        commands = [f"{command} --stage evaluate --split test"]
    else:
        commands = [f"{command} --stage calibrate"]
    source = ""
    if isinstance(config, SftExperimentConfig):
        source = (
            f"\nDataset input (read-only): `{config.data.dataset_from}`. Original split assignments and seeds are retained.\n"
            if config.data.dataset_from is not None
            else "\nDataset generation requires `data.generation.model` and `generations_per_task`.\n"
        )
    execution = (
        "\nEach execution allocates a numbered run and copies its config and task selections. "
        "Calibration retains partial results on failure; resumption is not supported.\n"
        if isinstance(config, CalibrationExperimentConfig)
        else "\nExecution of these model stages is not implemented.\n"
    )
    return (
        f"# {directory.name}\n\n"
        f"Method: `{config.experiment.method.value}`. Seed: `{config.experiment.seed}`.\n"
        f"Defaults copied from `{recorded_path(template_path)}`; `config.toml` owns the active values.\n"
        "Task files contain resolved IDs and are not resampled at execution.\n"
        f"Outputs: `outputs/{directory.name}/<run-number>/`.\n"
        + source
        + execution
        + "\nStage commands:\n\n```bash\n"
        + "\n".join(commands)
        + "\n```\n"
    )


def create_experiment(request: CreateRequest) -> Path:
    """Validate all inputs before creating a numbered directory; never execute work."""
    if NAME_PATTERN.fullmatch(request.name) is None:
        raise ValueError(
            "--name must contain lowercase letters/digits separated by single hyphens"
        )
    if request.dataset_from is not None and request.method != ExperimentMethod.SFT:
        raise ValueError("--dataset-from is only valid for SFT")
    if request.method == ExperimentMethod.CALIBRATE and (
        request.base_model_name_or_path is not None
        or request.base_model_revision is not None
        or request.adapter_path is not None
        or request.model_provider != ModelProvider.LOCAL
    ):
        raise ValueError("calibration does not accept model settings")
    template_path = latest_template(request.method)
    template = load_config(template_path)
    if template.experiment.method != request.method:
        raise ValueError(f"default template declares the wrong method: {template_path}")
    selected = resolve_selections(request)
    config = configure(request, template, selected)
    numbers = (
        [
            int(match["number"])
            for path in EXPERIMENTS_DIRECTORY.iterdir()
            if path.is_dir()
            if (match := re.match(r"(?P<number>[0-9]+)-", path.name)) is not None
        ]
        if EXPERIMENTS_DIRECTORY.exists()
        else []
    )
    number = max(numbers) + 1 if numbers else 0
    directory = (
        EXPERIMENTS_DIRECTORY / f"{number:0{EXPERIMENT_NUMBER_WIDTH}d}-{request.name}"
    )
    directory.mkdir(parents=True)
    try:
        (directory / "config.toml").write_text(
            tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)),
            encoding="utf-8",
        )
        for role, selection in selected.selections.items():
            (directory / TASK_FILES[role]).write_text(
                selection.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        (directory / "README.md").write_text(
            readme(directory, template_path, config), encoding="utf-8"
        )
        (directory / "run.py").write_text(RUN_SCRIPT, encoding="utf-8")
    except BaseException:
        shutil.rmtree(directory)
        raise
    return directory
