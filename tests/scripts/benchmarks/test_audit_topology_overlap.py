from pathlib import Path

import pytest

from qorl.taskset.taskset import TaskSet
from scripts.benchmarks import audit_topology_overlap as audit
from scripts.benchmarks.schemas import (
    JoinEdge,
    JoinTopology,
    RelationAlias,
    TableName,
    TopologyHash,
)


def sql_hash(sql: str) -> TopologyHash:
    return audit.topology_sha256(audit.extract_topology(sql, "test.sql"))


def test_topologies_ignore_alias_names_and_filter_literals() -> None:
    first = """
        SELECT COUNT(*)
        FROM title AS t, movie_info AS mi, info_type AS it
        WHERE t.id = mi.movie_id AND mi.info_type_id = it.id
          AND t.production_year > 1990
    """
    second = """
        SELECT COUNT(*)
        FROM title AS movie, movie_info AS fact, info_type AS kind
        WHERE movie.id = fact.movie_id AND fact.info_type_id = kind.id
          AND movie.production_year > 2005
    """
    assert sql_hash(first) == sql_hash(second)


def test_topology_ignores_join_columns() -> None:
    first = "SELECT COUNT(*) FROM title AS t, movie_info AS mi WHERE t.id = mi.movie_id"
    second = "SELECT COUNT(*) FROM title AS t, movie_info AS mi WHERE t.kind_id = mi.info_type_id"
    assert sql_hash(first) == sql_hash(second)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM info_type it, movie_info mi, title t WHERE t.id = mi.movie_id AND mi.info_type_id = it.id",
        "SELECT * FROM title t, movie_info mi, info_type it WHERE mi.info_type_id = it.id AND t.id = mi.movie_id",
        "SELECT * FROM title t, movie_info mi, info_type it WHERE mi.movie_id = t.id AND it.id = mi.info_type_id",
    ],
    ids=["from_order", "predicate_order", "equality_direction"],
)
def test_join_order_does_not_change_topology(sql: str) -> None:
    original = "SELECT * FROM title t, movie_info mi, info_type it WHERE t.id = mi.movie_id AND mi.info_type_id = it.id"
    assert sql_hash(original) == sql_hash(sql)


def test_repeated_table_instances_are_alias_independent() -> None:
    first = """
        SELECT COUNT(*)
        FROM info_type AS it1, movie_info AS mi, info_type AS it2
        WHERE it1.id = mi.info_type_id AND it2.id = mi.info_type_id
    """
    second = """
        SELECT COUNT(*)
        FROM info_type AS right_type, movie_info AS fact, info_type AS left_type
        WHERE left_type.id = fact.info_type_id AND right_type.id = fact.info_type_id
    """
    assert sql_hash(first) == sql_hash(second)


def test_topology_preserves_adjacency_and_table_names() -> None:
    a, b, c, d = map(RelationAlias, ("a", "b", "c", "d"))
    tables_by_alias = dict.fromkeys((a, b, c, d), TableName("title"))
    star = JoinTopology(
        tables_by_alias,
        {JoinEdge.between(a, b), JoinEdge.between(a, c), JoinEdge.between(a, d)},
    )
    chain = JoinTopology(
        tables_by_alias,
        {JoinEdge.between(a, b), JoinEdge.between(b, c), JoinEdge.between(c, d)},
    )
    different_tables = JoinTopology(
        {**tables_by_alias, d: TableName("name")}, chain.edges
    )

    assert audit.topology_sha256(star) != audit.topology_sha256(chain)
    assert audit.topology_sha256(chain) != audit.topology_sha256(different_tables)


def test_parallel_predicates_do_not_change_topology() -> None:
    first = "SELECT COUNT(*) FROM title AS t, movie_info AS mi WHERE t.id = mi.movie_id"
    second = first + " AND mi.info_type_id = t.kind_id AND mi.movie_id = t.id"
    assert sql_hash(first) == sql_hash(second)


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("SELECT 1", "FROM/WHERE"),
        ("SELECT * FROM title WHERE id = 1", "FROM term"),
        ("SELECT * FROM title t, name t WHERE t.id = t.id", "duplicate table alias"),
        ("SELECT * FROM title t, name n WHERE t.id = 1", "no usable join topology"),
        (
            "SELECT * FROM title t, name n, movie_info mi WHERE t.id = mi.movie_id",
            "disconnected",
        ),
    ],
    ids=["missing_from", "missing_alias", "duplicate_alias", "no_join", "disconnected"],
)
def test_rejects_unusable_sql(sql: str, error: str) -> None:
    with pytest.raises(RuntimeError, match=error):
        audit.extract_topology(sql, "test.sql")


def test_rejects_excessive_alias_permutations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "MAX_ALIAS_PERMUTATIONS", 1)
    a, b = RelationAlias("a"), RelationAlias("b")
    topology = JoinTopology(
        {a: TableName("title"), b: TableName("title")}, {JoinEdge.between(a, b)}
    )
    with pytest.raises(RuntimeError, match="too many alias permutations"):
        audit.topology_sha256(topology)


def test_reads_job_sql_without_an_inventory(repository_root: Path) -> None:
    topologies = audit.read_topologies(repository_root / "benchmarks/job/queries")
    assert sum(map(len, topologies.values())) == 113
    assert all(path.suffix == ".sql" for paths in topologies.values() for path in paths)


@pytest.mark.parametrize("benchmark", ["job", "ceb"])
def test_catalog_template_topologies_match_every_task(
    benchmark_task_sets: dict[str, TaskSet], benchmark: str
) -> None:
    task_set = benchmark_task_sets[benchmark]
    for task in task_set.tasks:
        topology = audit.extract_topology(task_set.load_sql(task), task.sql_path)
        assert (
            audit.topology_sha256(topology)
            == task_set.tasks_metadata[task.template_id].topology_sha256
        ), task.task_id


def test_groups_sql_files_with_matching_topologies(tmp_path: Path) -> None:
    first = tmp_path / "first.sql"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.sql"
    first.write_text("SELECT * FROM title t, movie_info mi WHERE t.id = mi.movie_id")
    second.write_text(
        "SELECT * FROM movie_info fact, title movie WHERE fact.movie_id = movie.id"
    )
    (tmp_path / "ignored.toml").write_text("not SQL or valid TOML")
    (tmp_path / "tasks.json").write_text("not a valid inventory")

    topologies = audit.read_topologies(tmp_path)

    assert len(topologies) == 1
    assert next(iter(topologies.values())) == [first, second]


@pytest.mark.parametrize("create_directory", [False, True], ids=["missing", "empty"])
def test_requires_sql_files(tmp_path: Path, create_directory: bool) -> None:
    directory = tmp_path / "queries"
    if create_directory:
        directory.mkdir()
    with pytest.raises(RuntimeError, match="no SQL files"):
        audit.read_topologies(directory)


@pytest.mark.parametrize("overlaps", [False, True], ids=["distinct", "matching"])
def test_main_reads_both_sql_directories_without_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    overlaps: bool,
) -> None:
    job = tmp_path / "job"
    ceb = tmp_path / "ceb"
    job.mkdir()
    ceb.mkdir()
    (job / "1a.sql").write_text(
        "SELECT * FROM title t, movie_info mi WHERE t.id = mi.movie_id"
    )
    ceb_sql = (
        "SELECT * FROM movie_info fact, title movie WHERE fact.movie_id = movie.id"
        if overlaps
        else "SELECT * FROM name n, cast_info ci WHERE ci.person_id = n.id"
    )
    (ceb / "example.sql").write_text(ceb_sql)
    monkeypatch.setattr(audit, "JOB_SQL_DIRECTORY", job)
    monkeypatch.setattr(audit, "CEB_SQL_DIRECTORY", ceb)

    audit.main()

    output = capsys.readouterr().out
    assert "JOB: 1 SQL files, 1 distinct topologies" in output
    assert "CEB: 1 SQL files, 1 distinct topologies" in output
    assert f"Shared topologies: {int(overlaps)}" in output
    if overlaps:
        assert "JOB: 1a.sql" in output
        assert "CEB: example.sql" in output
    assert sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
    ) == ["ceb", "ceb/example.sql", "job", "job/1a.sql"]
