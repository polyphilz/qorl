from datetime import UTC, datetime

from qorl.util.time import utc_now


def test_utc_now_returns_current_iso_timestamp() -> None:
    before = datetime.now(UTC)
    actual = datetime.fromisoformat(utc_now())
    after = datetime.now(UTC)

    assert actual.tzinfo is UTC
    assert before <= actual <= after
