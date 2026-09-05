from pathlib import Path

import pytest

from qorl.cli import parser


@pytest.mark.parametrize("command", ["calibrate", "run"])
def test_pool_config_is_explicit_or_left_to_the_environment(command: str) -> None:
    selected = "docker/worker_pool/configs/001-poolconf-2x16"
    assert parser().parse_args(
        [command, "--pool-config", selected]
    ).pool_config == Path(selected)
    assert parser().parse_args([command]).pool_config is None
