import pytest
from pydantic import ValidationError

from qorl.measure.schemas import CalibrationSettings, RolloutMeasurementSettings


@pytest.mark.parametrize("warmups,trials", [(1, 20), (5, 1)])
def test_calibration_requires_two_runs(warmups: int, trials: int) -> None:
    with pytest.raises(ValidationError):
        CalibrationSettings(
            max_warmup_runs=warmups, num_trials=trials, default_timeout_seconds=300
        )


def test_candidate_timeout_has_no_absolute_cap() -> None:
    settings = RolloutMeasurementSettings(
        default_warmups=1,
        default_measurements=1,
        paired_warmups=1,
        paired_measurements=3,
        default_timeout_seconds=300,
        candidate_timeout_floor_seconds=5,
        candidate_timeout_multiplier=3,
    )
    assert "candidate_timeout_cap_seconds" not in settings.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs"):
        RolloutMeasurementSettings.model_validate(
            {**settings.model_dump(), "candidate_timeout_cap_seconds": 120}
        )
