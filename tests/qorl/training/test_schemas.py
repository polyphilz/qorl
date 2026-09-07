import pytest
from pydantic import ValidationError

from qorl.training.schemas import TrainingRuntimeSettings


@pytest.fixture
def training_runtime() -> TrainingRuntimeSettings:
    return TrainingRuntimeSettings(
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
def test_training_runtime_rejects_unsupported_options(
    field: str, value: str, training_runtime: TrainingRuntimeSettings
) -> None:
    with pytest.raises(ValidationError, match=field):
        TrainingRuntimeSettings.model_validate(
            {**training_runtime.model_dump(), field: value}
        )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_training_runtime_supports_both_training_precisions(
    dtype: str, training_runtime: TrainingRuntimeSettings
) -> None:
    settings = TrainingRuntimeSettings.model_validate(
        {
            **training_runtime.model_dump(),
            "optimization_dtype": dtype,
            "reduce_dtype": dtype,
        }
    )
    assert settings.optimization_dtype == settings.reduce_dtype == dtype


@pytest.mark.parametrize("attention", ["auto", "flash_attention_4"])
def test_hf_implementation_rejects_custom_only_attention(
    attention: str, training_runtime: TrainingRuntimeSettings
) -> None:
    with pytest.raises(ValidationError, match=r"attention.*implementation"):
        TrainingRuntimeSettings.model_validate(
            {
                **training_runtime.model_dump(),
                "implementation": "hf",
                "attention": attention,
            }
        )
