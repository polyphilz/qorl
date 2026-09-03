from __future__ import annotations

from pathlib import Path

import pytest
from tests.qorl.sft.factories import sample

from qorl.sft.sample import sample_limit, sampling_summary
from qorl.sft.schemas import DatasetConfig, SamplingMode, load_record


def config() -> DatasetConfig:
    return load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )


def test_sampling_summary_deduplicates_fingerprints_per_task() -> None:
    summary = sampling_summary([sample(), sample(2)])

    assert summary.novel_candidates == 2
    assert summary.distinct_novel_fingerprints == 1
    assert summary.distinct_novel_fingerprint_yield == 0.5


def test_normal_and_default_best_sampling_caps_are_enforced() -> None:
    assert sample_limit(config(), SamplingMode.NORMAL, 5, 2) == 6
    assert sample_limit(config(), SamplingMode.DEFAULT_BEST, 7, 24) == 30
    with pytest.raises(RuntimeError, match="above its 6-sample cap"):
        sample_limit(config(), SamplingMode.NORMAL, 5, 3)
    with pytest.raises(RuntimeError, match="begin after the normal sample cap"):
        sample_limit(config(), SamplingMode.DEFAULT_BEST, 6, 1)
