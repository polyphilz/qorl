"""Shared helpers for preparing the recovered CEB queries."""

from __future__ import annotations

import hashlib
import itertools
import math
import re


SQL_START = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
SQL_FROM = re.compile(r"\bFROM\b", re.IGNORECASE)
SPLIT_SALT = "ceb-v1-template-split-v1"


def extract_sql_bytes(qrep: bytes) -> bytes:
    """Extract the single SQL string without interpreting pickle opcodes."""
    candidates: list[str] = []
    for offset, opcode in enumerate(qrep):
        if opcode == 0x8C and offset + 2 <= len(qrep):
            start = offset + 2
            length = qrep[offset + 1]
        elif opcode == 0x58 and offset + 5 <= len(qrep):
            start = offset + 5
            length = int.from_bytes(qrep[offset + 1 : offset + 5], "little")
        else:
            continue

        end = start + length
        if end > len(qrep):
            continue
        try:
            value = qrep[start:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if SQL_START.search(value) and SQL_FROM.search(value):
            candidates.append(value)

    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise ValueError(f"expected exactly one SQL string, found {len(unique)}")
    return unique[0].encode("utf-8")


def choose_validation_templates(
    query_counts: dict[str, int],
    fraction: float = 0.25,
    salt: str = SPLIT_SALT,
) -> tuple[str, ...]:
    """Choose a template-held-out split with query count nearest the target."""
    templates = sorted(query_counts)
    count = max(1, math.ceil(len(templates) * fraction))
    target = sum(query_counts.values()) * fraction

    def score(selected: tuple[str, ...]) -> tuple[float, str]:
        distance = abs(sum(query_counts[name] for name in selected) - target)
        tie_break = hashlib.sha256(
            f"{salt}:{','.join(selected)}".encode("utf-8")
        ).hexdigest()
        return distance, tie_break

    return min(itertools.combinations(templates, count), key=score)
