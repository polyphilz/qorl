from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_is_deterministic_balanced_and_new(
    load_experiment: Callable[[str], dict[str, Any]],
) -> None:
    builder = load_experiment("experiments/004-rl-run-v2/build_inventory.py")
    source = json.loads(builder["SOURCE"].read_text(encoding="utf-8"))
    pilot = json.loads(builder["PILOT"].read_text(encoding="utf-8"))
    actual = json.loads(builder["OUTPUT"].read_text(encoding="utf-8"))

    assert actual == builder["build"](source, pilot)
    selected = actual["splits"]["train"]
    prior_ids = {
        item["task_id"] for split in pilot["splits"].values() for item in split
    }
    assert len(selected) == 400
    assert len({item["task_id"] for item in selected}) == 400
    assert not ({item["task_id"] for item in selected} & prior_ids)
    assert sorted(actual["selection"]["template_quotas"].values()) == (
        [33] * 8 + [34] * 4
    )
    first_half_counts: dict[str, int] = {}
    for item in selected[:200]:
        template_id = item["template_id"]
        first_half_counts[template_id] = first_half_counts.get(template_id, 0) + 1
    assert set(first_half_counts.values()) == {16, 17}
    for index in range(0, len(selected), 4):
        batch = selected[index : index + 4]
        assert len({item["template_id"] for item in batch}) == 4


def test_final_run_is_pinned_to_its_inputs_and_limits() -> None:
    config = tomllib.loads(
        (ROOT / "experiments/004-rl-run-v2/train.toml").read_text(encoding="utf-8")
    )
    source = config["orchestrator"]["train"]["source"][0]
    concurrency = config["orchestrator"]["concurrency"]

    assert config["max_steps"] == 100
    assert config["seq_len"] == 20_480
    assert config["run"]["name"] == "rl-run-v2"
    assert config["output_dir"] == "outputs/rl"
    assert not config["clean"]
    assert config["model"]["name"] == "outputs/rl/rl-pilot-v1-merged"
    assert (
        config["env_vars"]["QORL_RL_TIMEOUT_MANIFEST"]
        == "experiments/004-rl-run-v2/timeouts.json"
    )
    assert (
        config["env_vars"]["QORL_RL_WORKER_POOL_CONFIG"]
        == "configs/postgres/training-pool-v1.json"
    )
    assert config["orchestrator"]["group_size"] == 4
    assert config["orchestrator"]["batch_size"] == 16
    assert config["orchestrator"]["max_off_policy_steps"] == 1
    assert (
        concurrency["initial_inflight"],
        concurrency["min_inflight"],
        concurrency["max_inflight"],
    ) == (4, 4, 4)
    assert source["serve"]["max_concurrent"] == 4
    assert source["serve"]["pool"]["num_workers"] == 1
    assert (
        source["env"]["taskset"]["selection"]
        == "experiments/004-rl-run-v2/selection.json"
    )
    assert source["env"]["agent"]["max_total_tokens"] == 18_432
    assert config["inference"]["vllm"]["max_model_len"] == 20_480
    assert config["inference"]["vllm"]["max_num_seqs"] == 4
    assert config["ckpt"] == {"interval": 5, "keep_last": 2, "keep_interval": 10}

    run_dir = Path(config["output_dir"]) / config["run"]["name"]
    assert run_dir == Path("outputs/rl/rl-run-v2")
    assert run_dir / "checkpoints" == Path("outputs/rl/rl-run-v2/checkpoints")


def test_checkpoint_evaluation_cadence_and_cohort_are_frozen() -> None:
    config = json.loads(
        (ROOT / "experiments/004-rl-run-v2/checkpoint-evaluation.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["selection"] == "experiments/003-rl-pilot-v1/selection.json"
    assert config["split"] == "validation"
    assert config["rollout_seeds"] == [
        2026083100,
        2026083101,
        2026083102,
        2026083103,
    ]
    assert config["checkpoint_steps"] == list(range(10, 101, 10))
    assert config["concurrency"] == 4


def test_spike_matches_four_rollouts_to_four_database_workers() -> None:
    config = tomllib.loads(
        (ROOT / "experiments/004-rl-run-v2/concurrency-spike.toml").read_text(
            encoding="utf-8"
        )
    )
    source = config["orchestrator"]["train"]["source"][0]
    concurrency = config["orchestrator"]["concurrency"]

    assert config["max_steps"] == 3
    assert config["orchestrator"]["group_size"] == 4
    assert config["orchestrator"]["batch_size"] == 16
    assert config["orchestrator"]["max_off_policy_steps"] == 1
    assert (
        concurrency["initial_inflight"],
        concurrency["min_inflight"],
        concurrency["max_inflight"],
    ) == (4, 4, 4)
    assert source["serve"]["max_concurrent"] == 4
    assert source["serve"]["pool"]["num_workers"] == 1
    assert config["inference"]["vllm"]["max_num_seqs"] == 4
    assert (
        source["env"]["taskset"]["selection"]
        == "experiments/004-rl-run-v2/selection.json"
    )
