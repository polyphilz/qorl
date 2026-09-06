import tomllib
from pathlib import Path

import pytest
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.utils.loaders import (
    environment_class,
    harness_class,
    taskset_class,
)

from qorl.training.environment import QorlEnvironment
from qorl.training.harness import QorlHarness, QorlHarnessConfig
from qorl.training.taskset import QorlTaskset, QorlTasksetConfig


@pytest.mark.parametrize(
    "config_path",
    [
        "experiments/002-rl-spike/train.toml",
        "experiments/003-rl-pilot-v1/train.toml",
        "experiments/004-rl-run-v2/train.toml",
        "experiments/004-rl-run-v2/concurrency-spike.toml",
    ],
)
def test_training_configs_resolve_qorl_plugins(
    repository_root: Path, config_path: str
) -> None:
    config = tomllib.loads((repository_root / config_path).read_text())
    source = config["orchestrator"]["train"]["source"][0]
    environment = EnvConfig.model_validate(source["env"])

    assert environment.id == "qorl"
    assert environment.taskset.id == "qorl"
    assert environment.agent.harness.id == "qorl"
    assert environment_class(environment.taskset.id, environment.id) is QorlEnvironment
    assert taskset_class(environment.taskset.id) is QorlTaskset
    assert harness_class(environment.agent.harness.id) is QorlHarness
    assert isinstance(environment.taskset, QorlTasksetConfig)
    assert isinstance(environment.agent.harness, QorlHarnessConfig)
