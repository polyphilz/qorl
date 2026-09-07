import pytest
from pydantic import ValidationError

from qorl.experiment.create import latest_template
from qorl.experiment.schemas import (
    EvaluationExperimentConfig,
    ExperimentMethod,
    ModelExperimentConfig,
    load_config,
)
from qorl.inference.schemas import ServingSettings
from qorl.model.schemas import LocalInferenceSettings


def test_serving_cannot_override_shared_context() -> None:
    config = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(config, EvaluationExperimentConfig)
    assert isinstance(config.inference, LocalInferenceSettings)
    with pytest.raises(ValidationError, match="Extra inputs"):
        ServingSettings.model_validate(
            {**config.inference.serving.model_dump(), "context_length": 100}
        )


@pytest.mark.parametrize(
    "method", [ExperimentMethod.SFT, ExperimentMethod.RL, ExperimentMethod.EVAL]
)
def test_serving_defaults_do_not_duplicate_dependency_version(
    method: ExperimentMethod,
) -> None:
    config = load_config(latest_template(method))
    assert isinstance(config, ModelExperimentConfig)
    assert isinstance(config.inference, LocalInferenceSettings)
    assert "vllm_version" not in config.inference.serving.model_dump()
    assert "request_timeout_seconds" not in config.inference.serving.model_dump()
    assert config.model.request_timeout_seconds == 300
    assert config.inference.serving.reasoning_parser == "qwen3"
