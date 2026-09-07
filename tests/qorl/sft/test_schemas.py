import pytest
from pydantic import ValidationError

from qorl.experiment.create import latest_template
from qorl.experiment.schemas import ExperimentMethod, SftExperimentConfig, load_config
from qorl.model.schemas import ModelProvider, ModelSettings
from qorl.sft.schemas import (
    GenerationSettings,
    PreparedDatasetManifest,
    SftTrainingSettings,
)


@pytest.mark.parametrize("batch_size,micro_batch_size", [(3, 2), (2, 4)])
def test_sft_requires_whole_microbatches(
    batch_size: int, micro_batch_size: int
) -> None:
    config = load_config(latest_template(ExperimentMethod.SFT))
    assert isinstance(config, SftExperimentConfig)
    with pytest.raises(ValidationError, match="divisible"):
        SftTrainingSettings.model_validate(
            {
                **config.training.model_dump(),
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
            }
        )


def test_sft_requires_at_least_one_dataloader_worker() -> None:
    config = load_config(latest_template(ExperimentMethod.SFT))
    assert isinstance(config, SftExperimentConfig)
    with pytest.raises(ValidationError, match="num_workers"):
        SftTrainingSettings.model_validate(
            {**config.training.model_dump(), "num_workers": 0}
        )


def test_generation_placeholders_do_not_choose_a_model_or_count() -> None:
    settings = GenerationSettings(model="FILL_ME_IN", generations_per_task="FILL_ME_IN")
    assert settings.model == settings.generations_per_task == "FILL_ME_IN"


def test_local_generator_is_rejected() -> None:
    model = ModelSettings(
        provider=ModelProvider.LOCAL,
        name_or_path="a/local-model",
        context_length=20_480,
    )
    with pytest.raises(ValidationError, match="OpenAI or Anthropic"):
        GenerationSettings(model=model, generations_per_task=1)


def test_dataset_reuse_needs_both_conversation_splits() -> None:
    with pytest.raises(ValidationError, match="validation"):
        PreparedDatasetManifest.model_validate(
            {"schema_version": 1, "format": "qorl-conversations", "tools": "tools.json"}
        )


@pytest.mark.parametrize(
    "provider,name",
    [
        (ModelProvider.OPENAI, "gpt-6-astra"),
        (ModelProvider.ANTHROPIC, "claude-fable-5-1"),
    ],
)
def test_supported_hosted_generators(provider: ModelProvider, name: str) -> None:
    model = ModelSettings(provider=provider, name_or_path=name, context_length=20_480)
    assert GenerationSettings(model=model, generations_per_task=1).model == model


def test_other_generator_models_are_not_silently_accepted() -> None:
    model = ModelSettings(
        provider=ModelProvider.OPENAI,
        name_or_path="another-model",
        context_length=20_480,
    )
    with pytest.raises(ValidationError, match="supports only"):
        GenerationSettings(model=model, generations_per_task=1)
