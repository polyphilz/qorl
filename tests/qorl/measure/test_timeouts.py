import pytest

from qorl.measure.schemas import RolloutMeasurementSettings
from qorl.measure.timeouts import candidate_timeout_ms


@pytest.mark.parametrize(
    "median,expected",
    [(10.0, 5_000), (30_000.0, 90_000), (120_000.0, 360_000), (299_000.0, 897_000)],
)
def test_candidate_timeout_has_no_absolute_cap(median: float, expected: int) -> None:
    settings = RolloutMeasurementSettings(
        default_warmups=1,
        default_measurements=1,
        paired_warmups=1,
        paired_measurements=3,
        default_timeout_seconds=300.0,
        candidate_timeout_floor_seconds=5.0,
        candidate_timeout_multiplier=3.0,
    )
    assert candidate_timeout_ms(median, settings) == expected
