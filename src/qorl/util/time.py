from datetime import UTC, datetime


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 timestamp."""
    return datetime.now(UTC).isoformat()
