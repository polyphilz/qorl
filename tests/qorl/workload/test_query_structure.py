from __future__ import annotations

import json
import unittest
from pathlib import Path

from qorl.workload.query_structure import (
    extract_join_structure,
    task_join_fingerprints,
)


ROOT = Path(__file__).resolve().parents[3]


class QueryStructureTest(unittest.TestCase):
    def test_fingerprints_ignore_alias_names_and_filter_literals(self) -> None:
        first = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM title AS t, movie_info AS mi, info_type AS it
            WHERE t.id = mi.movie_id
              AND mi.info_type_id = it.id
              AND t.production_year > 1990
            """,
            "first.sql",
        )
        second = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM title AS movie, movie_info AS fact, info_type AS kind
            WHERE movie.id = fact.movie_id
              AND fact.info_type_id = kind.id
              AND movie.production_year > 2005
            """,
            "second.sql",
        )
        self.assertEqual(task_join_fingerprints(first), task_join_fingerprints(second))

    def test_topology_ignores_columns_but_exact_graph_does_not(self) -> None:
        first = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM title AS t, movie_info AS mi
            WHERE t.id = mi.movie_id
            """,
            "first.sql",
        )
        second = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM title AS t, movie_info AS mi
            WHERE t.kind_id = mi.info_type_id
            """,
            "second.sql",
        )
        first_graph, first_topology = task_join_fingerprints(first)
        second_graph, second_topology = task_join_fingerprints(second)
        self.assertNotEqual(first_graph, second_graph)
        self.assertEqual(first_topology, second_topology)

    def test_repeated_table_instances_are_alias_independent(self) -> None:
        first = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM info_type AS it1, movie_info AS mi, info_type AS it2
            WHERE it1.id = mi.info_type_id
              AND it2.id = mi.info_type_id
            """,
            "first.sql",
        )
        second = extract_join_structure(
            """
            SELECT COUNT(*)
            FROM info_type AS right_type, movie_info AS fact, info_type AS left_type
            WHERE left_type.id = fact.info_type_id
              AND right_type.id = fact.info_type_id
            """,
            "second.sql",
        )
        self.assertEqual(task_join_fingerprints(first), task_join_fingerprints(second))

    def test_shared_code_preserves_checked_in_job_hash(self) -> None:
        inventory = json.loads(
            (ROOT / "data/job/tasks.json").read_text(encoding="utf-8")
        )
        task = inventory["tasks"][0]
        graph_hash, _topology_hash = task_join_fingerprints(task)
        self.assertEqual(graph_hash, task["join_graph_sha256"])
