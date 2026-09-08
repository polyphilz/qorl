import tomllib
from pathlib import Path

import pytest
from verifiers.v1.utils.loaders import (
    environment_class,
    harness_class,
    resolve_env_config,
    taskset_class,
)

from qorl.rl.environment import QorlEnvironment
from qorl.rl.harness import QorlHarness
from qorl.rl.schemas import (
    GrpoSettings,
    QorlEnvironmentConfig,
    QorlHarnessConfig,
    QorlTasksetConfig,
    RlSettings,
    ScalarRewardSettings,
)
from qorl.rl.tasks import QorlTaskset


@pytest.mark.parametrize("override_defaults", [False, True])
def test_training_configs_construct_qorl_environment(
    repository_root: Path, override_defaults: bool
) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    supplied = QorlHarnessConfig.model_validate(
        {
            key: defaults[key]
            for key in ("agent", "measurement", "rl", "model", "inference")
        }
    )
    if override_defaults:
        supplied = QorlHarnessConfig(
            model=supplied.model,
            inference=supplied.inference,
            agent=supplied.agent.model_copy(
                update={"candidate_attempts": supplied.agent.candidate_attempts + 1}
            ),
            measurement=supplied.measurement.model_copy(
                update={
                    "paired_measurements": supplied.measurement.paired_measurements + 1
                }
            ),
            rl=RlSettings(
                algorithm=GrpoSettings(type="grpo"),
                reward=ScalarRewardSettings(
                    invalid_attempt_penalty=1,
                    duplicate_attempt_penalty=0,
                    timeout_attempt_penalty=1,
                    no_valid_candidate_reward=0,
                ),
            ),
        )
        fallback = QorlHarnessConfig(id="qorl")
        assert supplied.agent != fallback.agent
        assert supplied.measurement != fallback.measurement
        assert supplied.rl != fallback.rl

    environment = resolve_env_config(
        {
            "id": "qorl",
            "taskset": {"id": "qorl"},
            "agent": {"harness": supplied.model_dump()},
        }
    )

    assert isinstance(environment, QorlEnvironmentConfig)
    assert environment.id == "qorl"
    assert environment.taskset.id == "qorl"
    assert environment.agent.harness.id == "qorl"
    assert environment_class(environment.taskset.id, environment.id) is QorlEnvironment
    assert taskset_class(environment.taskset.id) is QorlTaskset
    assert harness_class(environment.agent.harness.id) is QorlHarness
    assert isinstance(environment.taskset, QorlTasksetConfig)
    assert isinstance(environment.agent.harness, QorlHarnessConfig)
    assert environment.agent.harness == supplied
    assert environment.agent_harnesses() == {"agent": environment.agent.harness}

    constructed = QorlEnvironment(environment)
    assert constructed.config is environment
    assert isinstance(constructed.taskset, QorlTaskset)
    assert isinstance(constructed._harnesses["agent"], QorlHarness)
    assert constructed._harnesses["agent"].config == supplied

    assert environment.timeout.episode is None
    assert environment.timeout.finalize is None
    assert environment.agent.timeout.setup is None
    assert environment.agent.timeout.rollout is None
    assert environment.agent.timeout.finalize is None
