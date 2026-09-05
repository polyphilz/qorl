from __future__ import annotations

import pytest

from qorl.workload.ceb import extract_sql_bytes


def short_unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"\x8c" + bytes([len(encoded)]) + encoded


def unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"\x58" + len(encoded).to_bytes(4, "little") + encoded


class TestCebExtraction:
    def test_extracts_short_binunicode_without_interpreting_pickle(self) -> None:
        query = "SELECT COUNT(*)\nFROM title AS t WHERE t.id = 1"
        qrep = b"\x80\x04binary" + short_unicode("sql") + short_unicode(query)
        assert extract_sql_bytes(qrep) == query.encode()

    def test_extracts_binunicode_after_bogus_opcode_bytes(self) -> None:
        query = "SELECT t.id\nFROM title AS t WHERE t.id = 1"
        qrep = b"\x58\xff\xff\xff\xffnoise" + unicode(query)
        assert extract_sql_bytes(qrep) == query.encode()

    def test_requires_exactly_one_query(self) -> None:
        with pytest.raises(ValueError, match="found 0"):
            extract_sql_bytes(short_unicode("not SQL"))
        with pytest.raises(ValueError, match="found 2"):
            extract_sql_bytes(
                short_unicode("SELECT 1 FROM title")
                + short_unicode("SELECT 2 FROM title")
            )
