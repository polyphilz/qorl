"""Check a file against its expected byte size and SHA-256 checksum."""

from pathlib import Path

from qorl.util.hashing import sha256_file


def verify_file(
    target: Path, expected_size_in_bytes: int, expected_checksum: str
) -> None:
    if not target.is_file():
        raise RuntimeError(
            f"Expected a regular file at {target}; it is missing or is not a file."
        )
    actual_size = target.stat().st_size
    if actual_size != expected_size_in_bytes:
        raise RuntimeError(
            f"Size mismatch for {target}: expected {expected_size_in_bytes} bytes, "
            f"got {actual_size} bytes."
        )
    actual_checksum = sha256_file(target)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"SHA-256 mismatch for {target}: expected {expected_checksum}, "
            f"got {actual_checksum}."
        )
