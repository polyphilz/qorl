"""Stable purpose-specific seeds."""

import hashlib
import json

SEED_BYTES = 4


def derive_seed(seed: int, purpose: str, *identifiers: str) -> int:
    """Derive an unsigned 32-bit seed from an integer and ordered identifiers."""
    encoded = json.dumps([seed, purpose, *identifiers], separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:SEED_BYTES], "big")
