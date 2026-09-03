from __future__ import annotations

from pathlib import Path

from tests.qorl.sft.factories import measurement

from qorl.sft.measure import label_measurements
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
