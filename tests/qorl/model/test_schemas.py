import pytest
from pydantic import ValidationError

from qorl.model.schemas import ModelSettings, ModelWeightIndex, TrainerModelSettings


def test_model_identity_requires_context_and_known_provider() -> None:
    with pytest.raises(ValidationError):
        ModelSettings.model_validate({"provider": "made-up", "name_or_path": "model"})


def test_weight_index_requires_nonempty_shard_map() -> None:
    with pytest.raises(ValidationError):
        ModelWeightIndex(weight_map={})


@pytest.fixture
def trainer_model() -> TrainerModelSettings:
    return TrainerModelSettings(
        implementation="custom",
        attention="flash_attention_2",
        optimization_dtype="bfloat16",
        reduce_dtype="bfloat16",
        compile=False,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("implementation", "unsupported"),
        ("attention", "eager"),
        ("optimization_dtype", "float16"),
        ("reduce_dtype", "float16"),
    ],
)
def test_trainer_model_rejects_unsupported_options(
    field: str, value: str, trainer_model: TrainerModelSettings
) -> None:
    with pytest.raises(ValidationError, match=field):
        TrainerModelSettings.model_validate(
            {**trainer_model.model_dump(), field: value}
        )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_trainer_model_supports_both_training_precisions(
    dtype: str, trainer_model: TrainerModelSettings
) -> None:
    settings = TrainerModelSettings.model_validate(
        {
            **trainer_model.model_dump(),
            "optimization_dtype": dtype,
            "reduce_dtype": dtype,
        }
    )
    assert settings.optimization_dtype == settings.reduce_dtype == dtype


@pytest.mark.parametrize("attention", ["auto", "flash_attention_4"])
def test_hf_implementation_rejects_custom_only_attention(
    attention: str, trainer_model: TrainerModelSettings
) -> None:
    with pytest.raises(ValidationError, match=r"attention.*implementation"):
        TrainerModelSettings.model_validate(
            {
                **trainer_model.model_dump(),
                "implementation": "hf",
                "attention": attention,
            }
        )
