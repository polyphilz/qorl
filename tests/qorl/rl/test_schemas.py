import pytest
from pydantic import ValidationError

from qorl.experiment.create import latest_template
from qorl.experiment.schemas import ExperimentMethod, RlExperimentConfig, load_config
from qorl.rl.schemas import RlSettings, RlTrainingSettings, ScalarRewardSettings


def test_rl_requires_whole_groups() -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    with pytest.raises(ValidationError, match="divisible"):
        RlTrainingSettings.model_validate(
            {**config.training.model_dump(), "batch_size": 15}
        )


def test_rl_inflight_capacity_must_fit_a_whole_group() -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    with pytest.raises(ValidationError, match=r"max_inflight.*group_size"):
        RlTrainingSettings.model_validate(
            {**config.training.model_dump(), "group_size": 8, "max_inflight": 4}
        )


@pytest.mark.parametrize("max_inflight", [8, 9])
def test_rl_inflight_capacity_can_equal_or_exceed_group_size(
    max_inflight: int,
) -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    settings = RlTrainingSettings.model_validate(
        {
            **config.training.model_dump(),
            "group_size": 8,
            "max_inflight": max_inflight,
        }
    )
    assert settings.max_inflight == max_inflight


def test_anchored_grpo_rejects_unused_scalar_penalties() -> None:
    config = load_config(latest_template(ExperimentMethod.RL))
    assert isinstance(config, RlExperimentConfig)
    reward = ScalarRewardSettings(
        invalid_attempt_penalty=0.1,
        duplicate_attempt_penalty=0.05,
        timeout_attempt_penalty=0.1,
        no_valid_candidate_reward=-3.0,
    )
    with pytest.raises(ValidationError, match="does not consume"):
        RlSettings(algorithm=config.rl.algorithm, reward=reward)


def test_grpo_requires_scalar_reward_settings() -> None:
    with pytest.raises(ValidationError, match=r"requires rl\.reward"):
        RlSettings.model_validate({"algorithm": {"type": "grpo"}})
