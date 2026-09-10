import json
import os
import runpy
import shutil
import socket
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomli_w
from prime_rl.configs.trainer import CosineSchedulerConfig

from qorl.experiment import create
from qorl.experiment.schemas import (
    CalibrationExperimentConfig,
    CreateRequest,
    EvaluationExperimentConfig,
    ExperimentMethod,
    ModelExperimentConfig,
    RlExperimentConfig,
    SftExperimentConfig,
    load_config,
)
from qorl.model.schemas import (
    AstraInferenceSettings,
    LocalInferenceSettings,
    ModelPreset,
    ModelProvider,
    ModelSettings,
    ReasoningEffort,
)
from qorl.sft.schemas import PreparedDatasetManifest
from qorl.taskset.schemas import TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.training.schemas import CheckpointSettings

CUSTOM_MODEL_CONCURRENCY = 2
CUSTOM_RETRY_ATTEMPTS = 5
CUSTOM_TEMPERATURE = 0.7
CUSTOM_MAX_NUM_SEQS = 2
CUSTOM_LORA_RANK = 8
CUSTOM_LEARNING_RATE = 3e-5
CUSTOM_CHECKPOINT_INTERVAL = 7
CUSTOM_ASTRA_MAX_TOKENS = 16_384


@pytest.mark.parametrize("method", list(ExperimentMethod))
def test_creates_each_method(
    method: ExperimentMethod,
    creation_request: CreateRequest,
    experiment_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = Mock(side_effect=AssertionError("creation attempted execution"))
    monkeypatch.setattr(subprocess, "run", execution)
    monkeypatch.setattr(subprocess, "Popen", execution)
    monkeypatch.setattr(socket, "create_connection", execution)
    monkeypatch.setattr(TaskSet, "load_sql", execution)
    if method in (ExperimentMethod.EVAL, ExperimentMethod.CALIBRATE):
        creation_request = replace(creation_request, tasksets=("test=job[01:2]",))
    if method == ExperimentMethod.CALIBRATE:
        creation_request = replace(
            creation_request, base_model_name_or_path=None, base_model_revision=None
        )
    directory = create.create_experiment(replace(creation_request, method=method))
    assert directory == experiment_directory / "000-example"
    config = load_config(directory / "config.toml")
    assert config.experiment.method == method
    assert config.experiment.seed == 42
    assert config.postgres.path == creation_request.postgres_config
    assert config.pool.path == creation_request.pool_config / "poolconf.json"
    if not isinstance(config, CalibrationExperimentConfig):
        assert config.measurement.default_measurements == 3
    expected_files = {"config.toml", "README.md", "run.py"}
    if method in (ExperimentMethod.SFT, ExperimentMethod.RL):
        expected_files |= {"training-tasks.json", "validation-tasks.json"}
    else:
        expected_files.add("test-tasks.json")
    assert {path.name for path in directory.iterdir()} == expected_files
    assert "qorl experiment run" in (directory / "README.md").read_text()
    assert (directory / "run.py").read_text() == create.RUN_SCRIPT
    if isinstance(config, SftExperimentConfig):
        generation = config.data.generation
        assert generation is not None
        assert isinstance(generation.model, ModelSettings)
        assert generation.model.provider == ModelProvider.OPENAI
        assert generation.model.name_or_path == "gpt-6-astra"
        assert generation.generations_per_task == 1
        assert config.training.epochs == 1
    if isinstance(config, CalibrationExperimentConfig):
        assert (
            not {"model", "training", "agent", "inference", "resources"}
            & config.model_dump().keys()
        )
    execution.assert_not_called()


def test_optional_training_test_split(creation_request: CreateRequest) -> None:
    directory = create.create_experiment(
        replace(
            creation_request,
            tasksets=(*creation_request.tasksets, "test=job"),
            seed=73,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, RlExperimentConfig)
    assert config.data.test is not None
    assert config.data.test.selection_seed == 73
    selection = TaskSelection.model_validate_json(
        (directory / "test-tasks.json").read_bytes()
    )
    assert len(selection.task_ids) == 113


def test_creation_freezes_exclusions_without_changing_source(
    creation_request: CreateRequest, tmp_path: Path
) -> None:
    original = create.resolve_selections(creation_request)
    excluded = original.selections[create.TaskRole.TRAIN]
    source = tmp_path / "prior-training-tasks.json"
    source.write_text(excluded.model_dump_json(indent=2) + "\n")
    before = source.read_bytes()
    directory = create.create_experiment(
        replace(creation_request, exclude_tasks_from=(source,))
    )
    selected = TaskSelection.model_validate_json(
        (directory / "training-tasks.json").read_bytes()
    )
    assert len(selected.task_ids) == len(excluded.task_ids)
    assert not set(selected.task_ids) & set(excluded.task_ids)
    assert source.read_bytes() == before
    assert (
        TaskSelection.model_validate_json(
            (directory / "excluded-tasks-000.json").read_bytes()
        )
        == excluded
    )
    assert (
        TaskSelection.model_validate_json(
            (directory / "validation-tasks.json").read_bytes()
        )
        == original.selections[create.TaskRole.VALIDATION]
    )
    assert "--exclude-tasks-from" in (directory / "README.md").read_text()


def test_creation_rejects_exclusions_with_reused_dataset(
    creation_request: CreateRequest, dataset_artifact: Path
) -> None:
    with pytest.raises(ValueError, match="cannot change --dataset-from selections"):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.SFT,
                dataset_from=dataset_artifact,
                exclude_tasks_from=(Path("unused.json"),),
            )
        )


@pytest.mark.parametrize("test_expression", [(), ("test=job[01:2]",)])
def test_dataset_reuse_preserves_ids_order_seeds_and_source(
    test_expression: tuple[str, ...],
    creation_request: CreateRequest,
    dataset_artifact: Path,
) -> None:
    source_bytes = {p.name: p.read_bytes() for p in dataset_artifact.iterdir()}
    source = PreparedDatasetManifest.model_validate_json(source_bytes["manifest.json"])
    directory = create.create_experiment(
        replace(
            creation_request,
            method=ExperimentMethod.SFT,
            dataset_from=dataset_artifact,
            tasksets=test_expression,
            seed=99,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, SftExperimentConfig)
    assert config.data.dataset_from == dataset_artifact
    assert config.data.generation is None
    assert "generation" not in config.data.model_dump(exclude_none=True)
    assert config.experiment.seed == 99
    for filename, split in (
        ("training-tasks.json", source.training),
        ("validation-tasks.json", source.validation),
    ):
        actual = TaskSelection.model_validate_json((directory / filename).read_bytes())
        assert actual == split.selection
    assert config.data.training.selection_seed == source.training.selection_seed
    assert config.data.validation.selection_seed == source.validation.selection_seed
    assert config.data.imported_generation_seeds is not None
    assert (
        config.data.imported_generation_seeds.training
        == source.training.generation_seed
    )
    assert (
        config.data.imported_generation_seeds.validation
        == source.validation.generation_seed
    )
    assert {p.name: p.read_bytes() for p in dataset_artifact.iterdir()} == source_bytes
    assert not (directory / "dataset").exists()


@pytest.mark.parametrize(
    "expression",
    ["train=ceb[2a:1]", "validation=ceb[4a:1]", "test=ceb[2b:1]", "test=ceb[4a:1]"],
)
def test_reuse_rejects_replacement_or_overlapping_splits(
    expression: str,
    creation_request: CreateRequest,
    dataset_artifact: Path,
    experiment_directory: Path,
) -> None:
    with pytest.raises(
        ValueError if not expression.startswith("test") else RuntimeError
    ):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.SFT,
                dataset_from=dataset_artifact,
                tasksets=(expression,),
            )
        )
    assert not experiment_directory.exists()


def test_reuse_checks_overlap_within_imported_splits(
    creation_request: CreateRequest,
    dataset_artifact: Path,
) -> None:
    path = dataset_artifact / "manifest.json"
    manifest = PreparedDatasetManifest.model_validate_json(path.read_bytes())
    manifest = manifest.model_copy(update={"validation": manifest.training})
    path.write_text(manifest.model_dump_json())
    with pytest.raises(RuntimeError, match="topology overlap"):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.SFT,
                dataset_from=dataset_artifact,
                tasksets=(),
            )
        )


@pytest.mark.parametrize(
    "member", ["manifest.json", "training.jsonl", "validation.jsonl"]
)
def test_reuse_requires_artifact_members(
    member: str,
    creation_request: CreateRequest,
    dataset_artifact: Path,
) -> None:
    (dataset_artifact / member).unlink()
    with pytest.raises((ValueError, OSError)):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.SFT,
                dataset_from=dataset_artifact,
                tasksets=(),
            )
        )


def test_template_version_is_numeric_and_copied(
    creation_request: CreateRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    original = create.latest_template(ExperimentMethod.RL)
    low = defaults / "9-rl.toml"
    high = defaults / "10-rl.toml"
    shutil.copyfile(original, low)
    template = load_config(original)
    assert isinstance(template, RlExperimentConfig)
    changed = template.model_copy(
        update={"training": template.training.model_copy(update={"max_steps": 123})}
    )
    high.write_text(tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True)))
    os.utime(high, (1, 1))
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", defaults)
    assert create.latest_template(ExperimentMethod.RL) == high
    directory = create.create_experiment(creation_request)
    copied = load_config(directory / "config.toml")
    assert isinstance(copied, RlExperimentConfig)
    assert copied.training.max_steps == 123
    before = (directory / "config.toml").read_bytes()
    high.write_text("edited after creation")
    assert (directory / "config.toml").read_bytes() == before


def test_numbering_uses_existing_directories_without_overwriting(
    creation_request: CreateRequest,
    experiment_directory: Path,
) -> None:
    existing = experiment_directory / "009-existing"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep")
    (experiment_directory / "999-not-a-directory").write_text("ignore")
    (experiment_directory / "unrelated").mkdir()
    first = create.create_experiment(creation_request)
    second = create.create_experiment(creation_request)
    assert first.name == "010-example"
    assert second.name == "011-example"
    assert (existing / "keep.txt").read_text() == "keep"
    assert (first / "training-tasks.json").read_bytes() == (
        second / "training-tasks.json"
    ).read_bytes()


@pytest.mark.parametrize("method", [ExperimentMethod.SFT, ExperimentMethod.RL])
@pytest.mark.parametrize(
    "tasksets", [(), ("train=ceb[2a:1]",), ("validation=ceb[4a:1]",)]
)
def test_training_requires_both_roles(
    method: ExperimentMethod,
    tasksets: tuple[str, ...],
    creation_request: CreateRequest,
    experiment_directory: Path,
) -> None:
    with pytest.raises(ValueError, match="require train and validation"):
        create.create_experiment(
            replace(creation_request, method=method, tasksets=tasksets)
        )
    assert not experiment_directory.exists()


@pytest.mark.parametrize("method", [ExperimentMethod.EVAL, ExperimentMethod.CALIBRATE])
@pytest.mark.parametrize(
    "tasksets", [(), ("train=ceb[2a:1]",), ("train=ceb[2a:1]", "test=job")]
)
def test_nontraining_requires_only_test(
    method: ExperimentMethod,
    tasksets: tuple[str, ...],
    creation_request: CreateRequest,
) -> None:
    request = replace(creation_request, method=method, tasksets=tasksets)
    if method == ExperimentMethod.CALIBRATE:
        request = replace(
            request, base_model_name_or_path=None, base_model_revision=None
        )
    with pytest.raises(ValueError, match="require only a test"):
        create.create_experiment(request)


@pytest.mark.parametrize(
    "name", ["", "../escape", "Upper", "two words", "-prefix", "a--b", "a/"]
)
def test_bad_names_write_nothing(
    name: str, creation_request: CreateRequest, experiment_directory: Path
) -> None:
    with pytest.raises(ValueError, match="--name"):
        create.create_experiment(replace(creation_request, name=name))
    assert not experiment_directory.exists()


@pytest.mark.parametrize("revision", [None, "main", "latest", "123abc", "FILL_ME_IN"])
def test_remote_model_needs_immutable_revision(
    revision: str | None, creation_request: CreateRequest
) -> None:
    with pytest.raises(ValueError, match="40-character commit"):
        create.create_experiment(
            replace(creation_request, base_model_revision=revision)
        )


def test_adapter_evaluation_records_base_and_adapter(
    creation_request: CreateRequest, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"fixture weights")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "bias": "none",
                "r": 1,
                "lora_alpha": 1,
            }
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture adapter")
    directory = create.create_experiment(
        replace(
            creation_request,
            method=ExperimentMethod.EVAL,
            tasksets=("test=job[01:1]",),
            base_model_name_or_path=str(base),
            base_model_revision=None,
            adapter_path=adapter,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, EvaluationExperimentConfig)
    assert config.model.name_or_path == str(base)
    assert config.model.revision is None
    assert config.model.adapter_path == adapter
    assert set(base.iterdir()) == {base / "config.json", base / "model.safetensors"}


def test_adapter_only_directory_is_not_a_model(
    creation_request: CreateRequest, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="adapter-only"):
        create.create_experiment(
            replace(
                creation_request,
                base_model_name_or_path=str(adapter),
                base_model_revision=None,
            )
        )


@pytest.mark.parametrize(
    "provider,model",
    [
        (ModelProvider.OPENAI, "gpt-6-astra"),
        (ModelProvider.OPENROUTER, "qwen/qwen3.8-2.4t-a95b"),
    ],
)
def test_hosted_evaluation_does_not_copy_local_knobs(
    provider: ModelProvider,
    model: str,
    creation_request: CreateRequest,
) -> None:
    directory = create.create_experiment(
        replace(
            creation_request,
            method=ExperimentMethod.EVAL,
            tasksets=("test=job[01:1]",),
            model_provider=provider,
            base_model_name_or_path=model,
            base_model_revision=None,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, EvaluationExperimentConfig)
    assert config.model.provider == provider
    openrouter = provider == ModelProvider.OPENROUTER
    assert config.model.base_url == (
        "https://openrouter.ai/api/v1" if openrouter else "https://api.openai.com/v1"
    )
    assert config.model.request_timeout_seconds == 600
    assert config.model.api_key_env == (
        "OPENROUTER_API_KEY" if openrouter else "OPENAI_API_KEY"
    )
    assert config.resources is None
    assert config.inference.model_dump() == (
        {
            "max_tokens": 32768,
            "reasoning_effort": "medium",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": None,
            "send_seed": False,
            "provider": {"only": ["modal"], "allow_fallbacks": False},
        }
        if openrouter
        else {
            "max_tokens": 32768,
            "reasoning_effort": "medium",
            "reasoning_summary": "auto",
        }
    )


def test_creation_preserves_model_api_settings(creation_request: CreateRequest) -> None:
    config = load_config(create.latest_template(creation_request.method))
    assert isinstance(config, ModelExperimentConfig)
    template = config.model.model_copy(
        update={
            "base_url": "http://127.0.0.1:9000/v1",
            "request_timeout_seconds": 600,
            "api_key_env": "QORL_TEST_MODEL_KEY",
            "max_concurrent_requests": CUSTOM_MODEL_CONCURRENCY,
        }
    )
    model = create.model_settings(creation_request, template)
    assert model.base_url == template.base_url
    assert model.request_timeout_seconds == template.request_timeout_seconds
    assert model.api_key_env == template.api_key_env
    assert model.max_concurrent_requests == template.max_concurrent_requests


@pytest.mark.parametrize("provider", list(ModelProvider))
def test_creation_preserves_custom_model_concurrency(
    provider: ModelProvider,
    creation_request: CreateRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = tmp_path / "defaults"
    shutil.copytree(create.DEFAULTS_DIRECTORY, defaults)
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", defaults)
    if provider == ModelProvider.OPENAI:
        creation_request = replace(
            creation_request,
            method=ExperimentMethod.EVAL,
            tasksets=("test=job[01:1]",),
            model_provider=provider,
            base_model_name_or_path="gpt-6-astra",
            base_model_revision=None,
        )
        source = create.latest_config(defaults / "models", "gpt-6-astra")
        template = ModelPreset.model_validate(tomllib.loads(source.read_text()))
    else:
        source = create.latest_template(creation_request.method)
        template = load_config(source)
        assert isinstance(template, ModelExperimentConfig)
    changed = template.model_copy(
        update={
            "model": template.model.model_copy(
                update={"max_concurrent_requests": CUSTOM_MODEL_CONCURRENCY}
            )
        }
    )
    source.write_text(tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True)))
    directory = create.create_experiment(creation_request)
    config = load_config(directory / "config.toml")
    assert isinstance(config, ModelExperimentConfig)
    assert config.model.max_concurrent_requests == changed.model.max_concurrent_requests


@pytest.mark.parametrize(
    "method", [ExperimentMethod.SFT, ExperimentMethod.RL, ExperimentMethod.EVAL]
)
def test_creation_preserves_custom_nested_settings(
    method: ExperimentMethod,
    creation_request: CreateRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrency, retries, inference/serving, and nested training values are copied."""
    defaults = tmp_path / "defaults"
    shutil.copytree(create.DEFAULTS_DIRECTORY, defaults)
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", defaults)
    if method == ExperimentMethod.EVAL:
        creation_request = replace(creation_request, tasksets=("test=job[01:2]",))
    source = create.latest_template(method)
    template = load_config(source)
    assert isinstance(template, ModelExperimentConfig)
    assert isinstance(template.inference, LocalInferenceSettings)
    updates: dict[str, object] = {
        "model": template.model.model_copy(
            update={
                "max_concurrent_requests": CUSTOM_MODEL_CONCURRENCY,
                "retry": template.model.retry.model_copy(
                    update={"max_attempts": CUSTOM_RETRY_ATTEMPTS}
                ),
            }
        ),
        "inference": template.inference.model_copy(
            update={
                "temperature": CUSTOM_TEMPERATURE,
                "thinking": True,
                "serving": template.inference.serving.model_copy(
                    update={"max_num_seqs": CUSTOM_MAX_NUM_SEQS}
                ),
            }
        ),
    }
    if isinstance(template, (SftExperimentConfig, RlExperimentConfig)):
        updates["training"] = template.training.model_copy(
            update={
                "runtime": template.training.runtime.model_copy(
                    update={"compile": True}
                ),
                "lora": template.training.lora.model_copy(
                    update={"rank": CUSTOM_LORA_RANK}
                ),
                "optimizer": template.training.optimizer.model_copy(
                    update={
                        "lr": CUSTOM_LEARNING_RATE,
                        "scheduler": CosineSchedulerConfig.model_validate(
                            {"type": "cosine", "warmup_steps": 0}
                        ),
                    }
                ),
                "checkpoints": CheckpointSettings(
                    type="interval", interval=CUSTOM_CHECKPOINT_INTERVAL
                ),
            }
        )
    changed = template.model_copy(update=updates)
    changed = type(changed).model_validate(changed.model_dump())
    source.write_text(tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True)))
    directory = create.create_experiment(replace(creation_request, method=method))
    config = load_config(directory / "config.toml")
    assert isinstance(config, ModelExperimentConfig)
    assert config.model.max_concurrent_requests == CUSTOM_MODEL_CONCURRENCY
    assert config.model.retry == changed.model.retry
    assert config.inference == changed.inference
    assert isinstance(config.inference, LocalInferenceSettings)
    assert config.inference.serving.max_num_seqs == CUSTOM_MAX_NUM_SEQS
    document = tomllib.loads((directory / "config.toml").read_text())
    assert document["inference"]["serving"]["max_num_seqs"] == CUSTOM_MAX_NUM_SEQS
    assert not {"decoding", "serving", "lora", "optimizer", "checkpoints"} & set(
        document
    )
    if isinstance(config, (SftExperimentConfig, RlExperimentConfig)):
        assert isinstance(changed, (SftExperimentConfig, RlExperimentConfig))
        assert config.training.runtime == changed.training.runtime
        assert config.training.lora == changed.training.lora
        assert config.training.optimizer == changed.training.optimizer
        assert config.training.checkpoints == changed.training.checkpoints
        assert document["training"]["optimizer"]["scheduler"]["type"] == "cosine"
        assert "model" not in document["training"]


def test_astra_creation_preserves_preset_inference_without_serving(
    creation_request: CreateRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = tmp_path / "defaults"
    shutil.copytree(create.DEFAULTS_DIRECTORY, defaults)
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", defaults)
    source = create.latest_config(defaults / "models", "gpt-6-astra")
    preset = ModelPreset.model_validate(tomllib.loads(source.read_text()))
    assert isinstance(preset.inference, AstraInferenceSettings)
    changed = preset.model_copy(
        update={
            "model": preset.model.model_copy(
                update={
                    "retry": preset.model.retry.model_copy(
                        update={"max_attempts": CUSTOM_RETRY_ATTEMPTS}
                    )
                }
            ),
            "inference": preset.inference.model_copy(
                update={
                    "max_tokens": CUSTOM_ASTRA_MAX_TOKENS,
                    "reasoning_effort": ReasoningEffort.HIGH,
                }
            ),
        }
    )
    source.write_text(tomli_w.dumps(changed.model_dump(mode="json", exclude_none=True)))
    directory = create.create_experiment(
        replace(
            creation_request,
            method=ExperimentMethod.EVAL,
            tasksets=("test=job[01:1]",),
            model_provider=ModelProvider.OPENAI,
            base_model_name_or_path="gpt-6-astra",
            base_model_revision=None,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, EvaluationExperimentConfig)
    assert config.model.retry == changed.model.retry
    assert config.inference == changed.inference
    document = tomllib.loads((directory / "config.toml").read_text())
    assert document["inference"] == {
        "max_tokens": CUSTOM_ASTRA_MAX_TOKENS,
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
    }
    assert "resources" not in document


def test_failed_file_write_removes_only_its_new_directory(
    creation_request: CreateRequest,
    experiment_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = experiment_directory / "000-existing"
    old.mkdir(parents=True)
    (old / "keep").write_text("keep")
    monkeypatch.setattr(create, "readme", Mock(side_effect=OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        create.create_experiment(creation_request)
    assert list(experiment_directory.iterdir()) == [old]
    assert (old / "keep").read_text() == "keep"


def test_generated_script_delegates_without_stage_logic(
    creation_request: CreateRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qorl.experiment import run

    directory = create.create_experiment(creation_request)
    execute = Mock(return_value=0)
    monkeypatch.setattr(run, "main", execute)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(directory / "run.py"), run_name="__main__")
    assert error.value.code == 0
    execute.assert_called_once_with(directory)


@pytest.mark.parametrize(
    "method", [ExperimentMethod.RL, ExperimentMethod.EVAL, ExperimentMethod.CALIBRATE]
)
def test_dataset_flag_is_sft_only(
    method: ExperimentMethod, creation_request: CreateRequest, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="only valid for SFT"):
        create.create_experiment(
            replace(creation_request, method=method, dataset_from=tmp_path)
        )


def test_missing_model_is_rejected(
    creation_request: CreateRequest, experiment_directory: Path
) -> None:
    with pytest.raises(ValueError, match="base-model-name-or-path is required"):
        create.create_experiment(
            replace(creation_request, base_model_name_or_path=None)
        )
    assert not experiment_directory.exists()


def test_calibration_rejects_model_flags(creation_request: CreateRequest) -> None:
    with pytest.raises(ValueError, match="calibration does not accept model"):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.CALIBRATE,
                tasksets=("test=job",),
            )
        )


def test_hosted_models_cannot_be_training_bases(
    creation_request: CreateRequest,
) -> None:
    with pytest.raises(ValueError, match="standalone evaluation"):
        create.create_experiment(
            replace(
                creation_request,
                model_provider=ModelProvider.OPENAI,
                base_model_revision=None,
            )
        )


def test_training_does_not_accept_separate_adapter(
    creation_request: CreateRequest, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="adapter-path is only valid"):
        create.create_experiment(replace(creation_request, adapter_path=tmp_path))


@pytest.mark.parametrize("path_kind", ["postgres", "pool"])
def test_configuration_paths_must_exist(
    path_kind: str,
    creation_request: CreateRequest,
    tmp_path: Path,
    experiment_directory: Path,
) -> None:
    missing = tmp_path / "missing"
    request = (
        replace(creation_request, postgres_config=missing)
        if path_kind == "postgres"
        else replace(creation_request, pool_config=missing)
    )
    with pytest.raises((ValueError, OSError)):
        create.create_experiment(request)
    assert not experiment_directory.exists()


def test_imported_task_ids_are_resolved_against_catalog(
    creation_request: CreateRequest, dataset_artifact: Path
) -> None:
    path = dataset_artifact / "manifest.json"
    manifest = PreparedDatasetManifest.model_validate_json(path.read_bytes())
    invalid_selection = TaskSelection(
        benchmark_id=manifest.training.selection.benchmark_id, task_ids=["missing-task"]
    )
    manifest = manifest.model_copy(
        update={
            "training": manifest.training.model_copy(
                update={"selection": invalid_selection}
            )
        }
    )
    path.write_text(manifest.model_dump_json())
    with pytest.raises(RuntimeError, match="unknown selected task"):
        create.create_experiment(
            replace(
                creation_request,
                method=ExperimentMethod.SFT,
                dataset_from=dataset_artifact,
                tasksets=(),
            )
        )


def test_artifact_cannot_reference_outside_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "external.jsonl").write_text("[]")
    with pytest.raises(ValueError, match="external artifact file"):
        create.artifact_file(source, Path("../external.jsonl"))
