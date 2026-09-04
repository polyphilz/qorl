from __future__ import annotations

from pathlib import Path

from tests.qorl.sft.factories import measurement

from qorl.sft.measure import label_measurements, select_default_best_search_tasks
from qorl.sft.schemas import CandidateLabel, DatasetConfig, TaskLabel, load_record


def test_measurement_labels_use_conservative_repeated_bounds() -> None:
    records = [measurement("win", "p1", 1.16, 1.14)]
    records.extend(measurement("default", f"p{index}", 1.04) for index in range(2, 8))
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )

    labeled, labels = label_measurements(records, config)

    assert labeled[0].candidate_label == CandidateLabel.AMBIGUOUS
    assert labeled[0].score_interval is not None
    assert labeled[0].score_interval.lower == 1.14
    assert labeled[0].score_interval.upper == 1.16
    assert labels == {
        "win": TaskLabel.INSUFFICIENT_FINGERPRINTS,
        "default": TaskLabel.DEFAULT_BEST,
    }


def test_default_best_search_uses_measured_diversity_and_speed_ceiling() -> None:
    records = [
        measurement("eligible", "p1", 1.01),
        measurement("eligible", "p2", 1.09),
        measurement("too-fast", "p3", 1.01),
        measurement("too-fast", "p4", 1.11),
        measurement("too-few", "p5", 0.9),
    ]
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )

    labeled, _ = label_measurements(records, config)
    selected = select_default_best_search_tasks(labeled, config)

    assert selected == ["eligible"]
