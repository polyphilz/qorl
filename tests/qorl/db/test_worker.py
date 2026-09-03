from __future__ import annotations

import unittest

from qorl.db.exceptions import QueryTimeout, WorkerError
from qorl.db.worker import PostgresWorker


class TimeoutWorker(PostgresWorker):
    def __init__(self) -> None:
        self.explain_calls = 0
        self.explain_analyze_calls = 0

    @property
    def container(self) -> str:
        return "test"

    def execute(self, *args, **kwargs):
        raise WorkerError("ERROR: canceling statement due to statement timeout")


class WorkerTest(unittest.TestCase):
    def test_statement_timeout_has_a_specific_error_type(self) -> None:
        with self.assertRaises(QueryTimeout) as raised:
            TimeoutWorker().explain("SELECT 1", 5_000, analyze=True)

        self.assertEqual(raised.exception.timeout_ms, 5_000)
