"""Per-statement limits derived from measurement settings, not taskset calibration."""

import math

from qorl.measure.schemas import RolloutMeasurementSettings

MILLISECONDS_PER_SECOND = 1_000
DEFAULT_STATEMENT_TIMEOUT_MS = 300_000


def seconds_to_ms(seconds: float) -> int:
    """Round a positive statement limit up to whole milliseconds."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("statement timeout must be finite and positive")
    return math.ceil(seconds * MILLISECONDS_PER_SECOND)


def candidate_timeout_ms(
    default_median_ms: float, settings: RolloutMeasurementSettings
) -> int:
    """Derive the candidate cutoff from the initial baseline, with a floor but no cap."""
    if not math.isfinite(default_median_ms) or default_median_ms <= 0:
        raise ValueError("default median must be finite and positive")
    return max(
        seconds_to_ms(settings.candidate_timeout_floor_seconds),
        math.ceil(default_median_ms * settings.candidate_timeout_multiplier),
    )
