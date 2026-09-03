from __future__ import annotations

import json
import tomllib
from collections import Counter

from qorl.sft.assemble import select_tasks
from qorl.sft.schemas import (
    ActionFamily,
    DatasetConfig,
    load_json_object,
    load_record,
    require_list,
)


def test_selection_is_balanced_and_disjoint(repository_root, load_experiment) -> None:
    builder = load_experiment("experiments/005-protocol-sft-v2/build_inventory.py")
    source = load_json_object(repository_root / "data/ceb/tasks.json")
    selection = builder["build"](source)
    sampling = selection.splits.sampling
    live_gate = selection.splits.live_gate
    validation = selection.splits.validation

    assert len(sampling) == 300
    assert len(live_gate) == 64
    assert len(validation) == 64
    assert not (
        {item.task_id for item in sampling} & {item.task_id for item in live_gate}
    )
    assert set(Counter(item.template_id for item in sampling).values()) == {25}
    first_hundred = Counter(item.template_id for item in sampling[:100])
    assert max(first_hundred.values()) - min(first_hundred.values()) == 1
    assert selection.selection.rl_v3_exclusion_splits == [
        "sampling",
        "live_gate",
    ]
    assert {item.template_id for item in validation} == {
        "ceb-11b",
        "ceb-2b",
        "ceb-2c",
        "ceb-9a",
    }
    expected_validation = select_tasks(
        require_list(source.get("tasks"), "tasks"), "validation", 64, 20260830
    )
    assert [item.task_id for item in validation] == [
        item["task_id"] for item in expected_validation
    ]


def test_dataset_config_pins_v2_policy_and_thresholds(repository_root) -> None:
    config = load_record(
        repository_root / "experiments/005-protocol-sft-v2/dataset.json",
        DatasetConfig,
    )

    assert config.policy_config == "configs/policy/run-v2.json"
    assert config.sampling.concurrency == 4
    assert config.sampling.initial_samples_per_task == 4
    assert config.sampling.normal_maximum_samples_per_task == 6
    assert config.sampling.default_best_search_maximum_samples_per_task == 30
    assert config.sampling.fallback_check_task_count == 100
    assert config.sampling.fallback_yield_floor == 0.11
    assert config.labels.win_speedup == 1.15
    assert config.labels.default_best_maximum_speedup == 1.05
    assert config.split_counts.train == 256
    assert config.split_counts.validation == 64
    assert config.measurement.maximum_candidates_per_task == 6
    assert config.assembly.maximum_examples_per_task == 2
    assert config.training.validation_points == 4
    assert config.training.maximum_validation_interval == 15
    assert config.gate.samples_per_task == 4
    assert config.gate.concurrency == 4
    assert config.gate.constraint_satisfied_rate_floor == 0.9
    assert config.gate.default_duplicate_rate_ceiling == 0.05
    assert config.gate.novel_candidate_rate_improvement == 0.05
    assert config.gate.unlabeled_intervention_rate_floor == 0.5
    assert config.gate.fingerprints_per_intervened_task_floor == 2.0
    assert config.gate.action_family_rate_floor == 0.01
    assert set(config.gate.required_action_families) == set(ActionFamily)


def test_training_validation_interval_produces_several_loss_points(
    repository_root, load_experiment
) -> None:
    runner = load_experiment("experiments/005-protocol-sft-v2/run.py")
    config = load_record(
        repository_root / "experiments/005-protocol-sft-v2/dataset.json",
        DatasetConfig,
    )

    assert runner["validation_interval"](40, config) == 10
    assert runner["validation_interval"](60, config) == 15


def test_training_template_matches_the_sampling_context(repository_root) -> None:
    template = (
        repository_root / "experiments/005-protocol-sft-v2/train.toml.template"
    ).read_text()
    training = tomllib.loads(
        template.replace("__MAX_STEPS__", "50")
        .replace("__VALIDATION_INTERVAL__", "12")
        .replace("__SEQUENCE_LENGTH__", "20480")
        .replace("__DATASET_SEED__", "20260903")
    )
    policy = json.loads((repository_root / "configs/policy/run-v2.json").read_text())[
        "policy"
    ]

    assert training["model"]["seq_len"] == policy["context_length"] == 20_480
    assert training["renderer"] == {"name": "qwen3.5", "enable_thinking": False}
    assert training["model"]["lora"] == {"rank": 16, "alpha": 32, "dropout": 0.0}
