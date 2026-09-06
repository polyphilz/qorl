import json
import tomllib
from pathlib import Path

from qorl.agent.config import QoAgentConfig


def test_numbered_model_configs_differ_only_in_context_and_presence_penalty(
    repository_root: Path,
) -> None:
    root = repository_root / "model/configs"
    first = json.loads((root / "000-modelconf/modelconf.json").read_text())
    second = json.loads((root / "001-modelconf/modelconf.json").read_text())
    assert set(first) == set(second) == {"policy"}

    original = QoAgentConfig.from_dict(first["policy"])
    current = QoAgentConfig.from_dict(second["policy"])
    assert (original.context_length, original.sampling["presence_penalty"]) == (
        262_144,
        2.0,
    )
    assert (current.context_length, current.sampling["presence_penalty"]) == (
        20_480,
        0.0,
    )

    first["policy"]["context_length"] = second["policy"]["context_length"]
    first["policy"]["sampling"]["presence_penalty"] = second["policy"]["sampling"][
        "presence_penalty"
    ]
    assert first == second


def test_current_model_config_matches_vllm_dependency(repository_root: Path) -> None:
    root = repository_root / "model/configs"
    previous = json.loads((root / "001-modelconf/modelconf.json").read_text())
    current = json.loads((root / "002-modelconf/modelconf.json").read_text())
    project = tomllib.loads((repository_root / "pyproject.toml").read_text())

    config = QoAgentConfig.from_dict(current["policy"])
    assert f"vllm=={config.vllm_version}" in project["project"]["dependencies"]
    previous["policy"]["vllm_version"] = config.vllm_version
    assert previous == current
