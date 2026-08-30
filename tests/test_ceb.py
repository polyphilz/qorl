from __future__ import annotations

import unittest

from scripts.utils.ceb import choose_validation_templates, extract_sql_bytes


def short_unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"\x8c" + bytes([len(encoded)]) + encoded


def unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"\x58" + len(encoded).to_bytes(4, "little") + encoded


class CebExtractionTest(unittest.TestCase):
    def test_extracts_short_binunicode_without_interpreting_pickle(self) -> None:
        query = "SELECT COUNT(*)\nFROM title AS t WHERE t.id = 1"
        qrep = b"\x80\x04binary" + short_unicode("sql") + short_unicode(query)
        self.assertEqual(extract_sql_bytes(qrep), query.encode())

    def test_extracts_binunicode_after_bogus_opcode_bytes(self) -> None:
        query = "SELECT t.id\nFROM title AS t WHERE t.id = 1"
        qrep = b"\x58\xff\xff\xff\xffnoise" + unicode(query)
        self.assertEqual(extract_sql_bytes(qrep), query.encode())

    def test_requires_exactly_one_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            extract_sql_bytes(short_unicode("not SQL"))
        with self.assertRaisesRegex(ValueError, "found 2"):
            extract_sql_bytes(
                short_unicode("SELECT 1 FROM title")
                + short_unicode("SELECT 2 FROM title")
            )

    def test_template_split_balances_queries_and_is_deterministic(self) -> None:
        counts = {"a": 10, "b": 20, "c": 30, "d": 40}
        self.assertEqual(choose_validation_templates(counts), ("c",))


if __name__ == "__main__":
    unittest.main()
