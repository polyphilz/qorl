import json
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
