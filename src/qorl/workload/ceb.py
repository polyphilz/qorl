"""Shared helpers for preparing the recovered CEB queries."""

from __future__ import annotations

import re

SQL_START = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
SQL_FROM = re.compile(r"\bFROM\b", re.IGNORECASE)
SHORT_BINUNICODE_OPCODE = 0x8C
BINUNICODE_OPCODE = 0x58
SHORT_BINUNICODE_HEADER_BYTES = 2
BINUNICODE_HEADER_BYTES = 5


def extract_sql_bytes(qrep: bytes) -> bytes:
    """Extract the single SQL string without interpreting pickle opcodes."""
    candidates: list[str] = []
    for offset, opcode in enumerate(qrep):
        if (
            opcode == SHORT_BINUNICODE_OPCODE
            and offset + SHORT_BINUNICODE_HEADER_BYTES <= len(qrep)
        ):
            start = offset + SHORT_BINUNICODE_HEADER_BYTES
            length = qrep[offset + 1]
        elif opcode == BINUNICODE_OPCODE and offset + BINUNICODE_HEADER_BYTES <= len(
            qrep
        ):
            start = offset + BINUNICODE_HEADER_BYTES
            length = int.from_bytes(
                qrep[offset + 1 : offset + BINUNICODE_HEADER_BYTES], "little"
            )
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
