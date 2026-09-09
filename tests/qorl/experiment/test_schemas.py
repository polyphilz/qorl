from dataclasses import replace
from pathlib import Path

import pytest
import tomli_w
from prime_rl.configs.algorithm import QorlAnchoredGRPOAlgoConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    LinearSchedulerConfig,
    SchedulerConfig,
)
from pydantic import ValidationError

from qorl.experiment import create
from qorl.experiment.create import latest_template
from qorl.experiment.schemas import (
    PLACEHOLDER,
    CalibrationExperimentConfig,
    CreateRequest,
    ExperimentMethod,
    ModelExperimentConfig,
    ResourceSettings,
    RlExperimentConfig,
    SftExperimentConfig,
    load_config,
)
from qorl.training.schemas import CheckpointSettings, OptimizerSettings


@pytest.mark.parametrize("method", list(ExperimentMethod))
def test_default_templates_roundtrip_without_implicit_identities(
    method: ExperimentMethod, tmp_path: Path
) -> None:
    config = load_config(latest_template(method))
    assert config.experiment.method == method
    assert config.experiment.seed == 42
    assert config.postgres.path == Path(PLACEHOLDER)
    assert config.pool.path == Path(PLACEHOLDER)
    if isinstance(config, ModelExperimentConfig):
        assert config.model.name_or_path == PLACEHOLDER
        assert config.model.revision == PLACEHOLDER
        assert config.model.context_length == 32_768
        assert config.model.base_url == "http://127.0.0.1:8000/v1"
        assert config.model.request_timeout_seconds == 300
        assert config.model.api_key_env is None
        assert config.agent.candidate_attempts == 1
        assert config.measurement.default_timeout_seconds == 300
        assert config.measurement.default_warmups == 1
        assert config.measurement.default_measurements == 1
        assert config.measurement.paired_warmups == 1
        assert config.measurement.paired_measurements == 3
        assert config.measurement.candidate_timeout_floor_seconds == 5
        assert config.measurement.candidate_timeout_multiplier == 3
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))
    assert load_config(path) == config


@pytest.mark.parametrize("field", ["base_url", "request_timeout_seconds"])
def test_local_experiment_requires_model_api_settings(field: str) -> None:
    config = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(config, ModelExperimentConfig)
    with pytest.raises(ValidationError, match="local models require model"):
        type(config).model_validate(
            {
                **config.model_dump(),
                "model": {**config.model.model_dump(), field: None},
            }
        )


@pytest.mark.parametrize(
    "method,learning_rate", [(ExperimentMethod.SFT, 1e-4), (ExperimentMethod.RL, 1e-6)]
)
def test_training_defaults_match_pinned_trainer(
    method: ExperimentMethod, learning_rate: float
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, (SftExperimentConfig, RlExperimentConfig))
    inherited = AdamWConfig()
    assert config.training.optimizer.lr == learning_rate
    assert config.training.optimizer.weight_decay == 0
    assert config.training.max_grad_norm == inherited.max_norm
    assert config.training.optimizer.betas1 == inherited.betas1
    assert config.training.optimizer.betas2 == inherited.betas2
    assert config.training.optimizer.scheduler == ConstantSchedulerConfig()
    assert config.training.lora.rank == 16
    assert config.training.lora.alpha == 32
    assert config.training.lora.dropout == 0
    assert config.training.lora.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if isinstance(config, SftExperimentConfig):
        assert config.training.checkpoints.type == "final"
        assert config.training.batch_size == 1
    else:
        assert config.training.batch_size == 16
        assert config.training.group_size == 4
        assert config.training.max_steps == 100
        assert config.training.checkpoints.interval == 5
        assert config.training.checkpoints.keep_last == 2
        assert config.training.checkpoints.keep_interval == 10
        inherited_algorithm = QorlAnchoredGRPOAlgoConfig()
        assert config.rl.algorithm.model_dump() == inherited_algorithm.model_dump(
            exclude={"sampling", "expected_group_size"}
        )
        assert config.rl.reward is None
        assert "expected_group_size" not in config.rl.algorithm.model_dump()


@pytest.mark.parametrize(
    "method", [ExperimentMethod.SFT, ExperimentMethod.RL, ExperimentMethod.EVAL]
)
@pytest.mark.parametrize(
    "section", ["decoding", "serving", "lora", "optimizer", "checkpoints"]
)
def test_relocated_sections_have_no_top_level_aliases(
    method: ExperimentMethod, section: str
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, ModelExperimentConfig)
    document = config.model_dump()
    if section == "decoding":
        moved = document["inference"]
    elif section == "serving":
        moved = document["inference"]["serving"]
    elif "training" in document:
        moved = document["training"][section]
    else:
        pytest.skip(f"{method.value} templates have no {section} settings")
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(config).model_validate({**document, section: moved})


@pytest.mark.parametrize("method", [ExperimentMethod.SFT, ExperimentMethod.RL])
def test_training_runtime_section_replaces_training_model(
    method: ExperimentMethod,
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, (SftExperimentConfig, RlExperimentConfig))
    training = config.training.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(config.training).model_validate({**training, "model": training["runtime"]})


def test_unknown_or_irrelevant_sections_are_rejected() -> None:
    config = load_config(latest_template(ExperimentMethod.CALIBRATE))
    assert isinstance(config, CalibrationExperimentConfig)
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(config).model_validate(
            {**config.model_dump(), "model": {"name": "ignored"}}
        )


@pytest.mark.parametrize("method", [ExperimentMethod.SFT, ExperimentMethod.RL])
@pytest.mark.parametrize("limit", [None, 0.0, 1.0, 2.5])
def test_gradient_clipping_survives_toml_roundtrip(
    method: ExperimentMethod, limit: float | None, tmp_path: Path
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, (SftExperimentConfig, RlExperimentConfig))
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_grad_norm": limit})}
    )
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))
    assert load_config(path) == config


@pytest.mark.parametrize("method", [ExperimentMethod.SFT, ExperimentMethod.RL])
@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_gradient_clipping_rejects_negative_or_nonfinite_limits(
    method: ExperimentMethod, limit: float
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, (SftExperimentConfig, RlExperimentConfig))
    with pytest.raises(ValidationError, match="max_grad_norm"):
        type(config.training).model_validate(
            {**config.training.model_dump(), "max_grad_norm": limit}
        )


def test_optimizer_defaults_match_native_hyperparameters() -> None:
    optimizer = OptimizerSettings(scheduler=ConstantSchedulerConfig())
    assert optimizer.model_dump(exclude={"scheduler"}) == AdamWConfig.model_validate(
        {}
    ).model_dump(exclude={"max_norm"})


@pytest.mark.parametrize("field", ["max_norm", "max_grad_norm"])
def test_optimizer_does_not_own_gradient_clipping(field: str) -> None:
    optimizer = OptimizerSettings(scheduler=ConstantSchedulerConfig())
    with pytest.raises(ValidationError, match="Extra inputs"):
        OptimizerSettings.model_validate({**optimizer.model_dump(), field: 1.0})


@pytest.mark.parametrize("method", [ExperimentMethod.SFT, ExperimentMethod.RL])
@pytest.mark.parametrize("limit", [None, 2.5])
def test_creation_preserves_configured_gradient_clipping(
    method: ExperimentMethod,
    limit: float | None,
    creation_request: CreateRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = latest_template(method)
    config = load_config(source)
    assert isinstance(config, (SftExperimentConfig, RlExperimentConfig))
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_grad_norm": limit})}
    )
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / source.name).write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    monkeypatch.setattr(create, "DEFAULTS_DIRECTORY", defaults)
    directory = create.create_experiment(replace(creation_request, method=method))
    actual = load_config(directory / "config.toml")
    assert isinstance(actual, (SftExperimentConfig, RlExperimentConfig))
    assert actual.training.max_grad_norm == limit


@pytest.mark.parametrize(
    "updates",
    [
        {"type": "interval"},
        {"type": "final", "interval": 5},
        {"type": "final", "keep_last": 2},
    ],
)
def test_checkpoint_schedule_has_no_ignored_settings(
    updates: dict[str, str | int],
) -> None:
    with pytest.raises(ValidationError):
        CheckpointSettings.model_validate(updates)


def test_rl_gpu_groups_cannot_overlap() -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    with pytest.raises(ValidationError, match="disjoint"):
        type(config).model_validate(
            {
                **config.model_dump(),
                "resources": ResourceSettings(
                    training_gpu_ids=[0], serving_gpu_ids=[0]
                ),
            }
        )


@pytest.mark.parametrize("batch_size,micro_batch_size", [(3, 1), (6, 2), (2, 2)])
def test_sft_batch_must_fit_whole_microbatches_across_training_gpus(
    batch_size: int, micro_batch_size: int, tmp_path: Path
) -> None:
    config = load_config(latest_template(ExperimentMethod.SFT))
    assert isinstance(config, SftExperimentConfig)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"batch_size": batch_size, "micro_batch_size": micro_batch_size}
            ),
            "resources": ResourceSettings(training_gpu_ids=[0, 1], serving_gpu_ids=[0]),
        }
    )
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))
    with pytest.raises(ValidationError, match=r"batch_size.*training GPU count"):
        load_config(path)


@pytest.mark.parametrize(
    "batch_size,micro_batch_size,training_gpu_ids",
    [(1, 1, [0]), (2, 1, [0, 1]), (4, 2, [0, 1]), (8, 2, [0, 1])],
)
def test_sft_accepts_distributed_batches_with_whole_microbatches(
    batch_size: int, micro_batch_size: int, training_gpu_ids: list[int]
) -> None:
    config = load_config(latest_template(ExperimentMethod.SFT))
    assert isinstance(config, SftExperimentConfig)
    validated = SftExperimentConfig.model_validate(
        {
            **config.model_dump(),
            "training": {
                **config.training.model_dump(),
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
            },
            "resources": ResourceSettings(
                training_gpu_ids=training_gpu_ids, serving_gpu_ids=[0]
            ),
        }
    )
    assert validated.training.batch_size == batch_size


@pytest.mark.parametrize(
    "scheduler",
    [
        pytest.param(CosineSchedulerConfig(warmup_steps=200), id="cosine-too-long"),
        pytest.param(CosineSchedulerConfig(warmup_steps=100), id="cosine-no-decay"),
        pytest.param(
            LinearSchedulerConfig(warmup_steps=60, decay_steps=50),
            id="linear-phases-too-long",
        ),
        pytest.param(
            LinearSchedulerConfig(warmup_steps=0, decay_steps=0), id="linear-no-phases"
        ),
    ],
)
def test_rl_rejects_scheduler_phases_that_do_not_fit_training(
    scheduler: SchedulerConfig, tmp_path: Path
) -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "optimizer": config.training.optimizer.model_copy(
                        update={"scheduler": scheduler}
                    )
                }
            )
        }
    )
    path = tmp_path / "config.toml"
    path.write_text(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))
    with pytest.raises(ValidationError, match="warmup_steps"):
        load_config(path)


@pytest.mark.parametrize(
    "scheduler",
    [
        ConstantSchedulerConfig(),
        CosineSchedulerConfig(warmup_steps=0),
        CosineSchedulerConfig(warmup_steps=99),
        LinearSchedulerConfig(warmup_steps=50, decay_steps=50),
    ],
)
def test_rl_accepts_scheduler_phases_that_fit_training(
    scheduler: SchedulerConfig,
) -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    validated = RlExperimentConfig.model_validate(
        {
            **config.model_dump(),
            "training": config.training.model_copy(
                update={
                    "optimizer": config.training.optimizer.model_copy(
                        update={"scheduler": scheduler}
                    )
                }
            ),
        }
    )
    assert validated.training.optimizer.scheduler == scheduler
