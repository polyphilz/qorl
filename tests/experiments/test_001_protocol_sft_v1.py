from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_and_training_configs_use_the_same_seed() -> None:
    experiment = ROOT / "experiments/001-protocol-sft-v1"
    dataset = json.loads((experiment / "dataset.json").read_text(encoding="utf-8"))
    training = tomllib.loads((experiment / "train.toml").read_text(encoding="utf-8"))

    assert dataset["split_counts"] == {"train": 256, "validation": 64}
    assert dataset["seed"] == training["data"]["seed"]
    assert dataset["seed"] == training["val"]["data"]["seed"]
