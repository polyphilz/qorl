from __future__ import annotations

import json
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qorl.util.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_pilot_config_has_twelve_four_group_updates(
    load_experiment: Callable[[str], dict[str, Any]],
) -> None:
    experiment = load_experiment("experiments/003-rl-pilot-v1/run.py")
    config = tomllib.loads((ROOT / experiment["CONFIG"]).read_text(encoding="utf-8"))

    assert config["max_steps"] == 12
    assert config["orchestrator"]["batch_size"] == 16
    assert config["orchestrator"]["group_size"] == 4
    assert (
        config["orchestrator"]["train"]["source"][0]["env"]["taskset"]["split"]
        == "train"
    )


def test_paired_validation_cohort_and_seeds_are_frozen() -> None:
    config = json.loads(
        (ROOT / "experiments/003-rl-pilot-v1/validation.json").read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads((ROOT / config["selection"]).read_text(encoding="utf-8"))
    selected = selection["splits"][config["split"]]

    assert len(selected) == 16
    assert len({item["task_id"] for item in selected}) == 16
    assert set(Counter(item["template_id"] for item in selected).values()) == {4}
    assert len({item["template_id"] for item in selected}) == 4
    assert config["rollout_seeds"] == [
        2026083100,
        2026083101,
        2026083102,
        2026083103,
    ]
    assert config["selection"] == "experiments/003-rl-pilot-v1/selection.json"
    assert config["run_config"] == "configs/policy/run-v1.json"


def test_pre_rl_gate_checks_inputs_model_and_reward_variance(
    tmp_path: Path,
    load_experiment: Callable[[str], dict[str, Any]],
) -> None:
    experiment = load_experiment("experiments/003-rl-pilot-v1/run.py")
    inputs = {
        "config_sha256": tmp_path / "experiments/003-rl-pilot-v1/validation.json",
        "selection_sha256": tmp_path / "experiments/003-rl-pilot-v1/selection.json",
        "run_config_sha256": tmp_path / "configs/policy/run-v1.json",
    }
    for path in inputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input")
    report = {
        "status": "completed",
        "phase": "pre",
        **{name: sha256_file(path) for name, path in inputs.items()},
        "model": {"model_safetensors_sha256": "model"},
        "summary": {
            "completed_rollout_count": 64,
            "orchestration_failure_count": 0,
            "task_group_count": 16,
            "nonzero_reward_variance_group_count": 16,
        },
    }
    path = tmp_path / experiment["PRE_RL_REPORT"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))

    experiment["verify_pre_rl_validation"](tmp_path, "model")
    report["summary"]["nonzero_reward_variance_group_count"] = 15
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="did not pass"):
        experiment["verify_pre_rl_validation"](tmp_path, "model")


def test_inventory_is_deterministic_and_template_balanced(
    load_experiment: Callable[[str], dict[str, Any]],
) -> None:
    builder = load_experiment("experiments/003-rl-pilot-v1/build_inventory.py")
    source = json.loads(builder["SOURCE"].read_text(encoding="utf-8"))
    expected = builder["build"](source)
    actual = json.loads(builder["OUTPUT"].read_text(encoding="utf-8"))

    assert actual == expected
    assert actual["counts"] == {"spike": 1, "train": 48, "validation": 16}
    assert {item["template_id"] for item in actual["splits"]["train"]} == {
        task["template_id"] for task in source["tasks"] if task["partition"] == "train"
    }
    assert {item["template_id"] for item in actual["splits"]["validation"]} == {
        task["template_id"]
        for task in source["tasks"]
        if task["partition"] == "validation"
    }
