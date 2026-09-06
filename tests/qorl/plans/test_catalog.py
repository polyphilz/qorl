from __future__ import annotations

import pytest

from qorl.plans.catalog import TaskCatalog
from qorl.postgres.schemas import PostgresIndexes
from qorl.taskset.schemas import Relation, Task


@pytest.mark.parametrize("aliases", [("t",), ("movie",), ("t1", "t2")])
def test_catalog_selects_only_referenced_tables(aliases: tuple[str, ...]) -> None:
    indexes = PostgresIndexes(
        by_table={
            "title": frozenset({"title_pkey", "kind_id_title"}),
            "unrelated": frozenset({"unrelated_pkey"}),
        }
    )
    task = Task(
        task_id="test",
        template_id="test",
        sql_path="test.sql",
        sql_sha256="unused",
        tables=["title", "no_indexes"],
        relations=[Relation(alias=alias, table="title") for alias in aliases]
        + [Relation(alias="n", table="no_indexes")],
        join_edges=[],
        table_count=2,
        relation_count=len(aliases) + 1,
        join_predicate_count=0,
    )

    catalog = TaskCatalog.from_postgres(task, indexes)
    assert catalog.relations == frozenset((*aliases, "n"))
    assert catalog.adjacency == dict.fromkeys((*aliases, "n"), frozenset())
    assert catalog.indexes == {
        **dict.fromkeys(aliases, frozenset({"title_pkey", "kind_id_title"})),
        "n": frozenset(),
    }
    catalog.indexes.clear()
    assert indexes.by_table["title"] == frozenset({"title_pkey", "kind_id_title"})
    assert indexes.by_table["unrelated"] == frozenset({"unrelated_pkey"})
