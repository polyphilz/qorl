import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.resources import load_runtime_profile
from qorl.util.hashing import sha256_bytes
from scripts.imdb import load_verify_archive as load
from scripts.imdb.schemas import ImdbManifest, ImdbMember


@pytest.mark.parametrize(
    ("start", "end", "expected_sha256"),
    [
        (
            "CREATE TABLE ",
            "\\copy ",
            "1d10ba0c9a881e890aeb951c0960cc4e3472c9c9695ce1edb6493e9e4466d3d2",
        ),
        (
            "create index ",
            "GRANT SELECT ",
            "727909bbbb9b0ae2b7d5739cfcd066992fc4bf1e984b29d87bcc700c4e2583cc",
        ),
    ],
    ids=["schema.sql", "fkindexes.sql"],
)
def test_load_sql_vendors_upstream_definitions_unchanged(
    repository_root: Path, start: str, end: str, expected_sha256: str
) -> None:
    sql = (repository_root / "scripts/imdb/load.sql").read_text()
    block = sql[sql.index(start) : sql.index(end)].rstrip() + "\n"
    assert sha256_bytes(block.encode()) == expected_sha256
    assert "\\ir " not in sql


def test_module_entrypoint(repository_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-E", "-m", "scripts.imdb.load_verify_archive", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("verification_fails", [False, True])
def test_load_verifies_then_stops_and_archives_without_fetching_or_restoring(
    repository_root: Path, tmp_path: Path, monkeypatch, verification_fails: bool
) -> None:
    (tmp_path / "docker").symlink_to(
        repository_root / "docker", target_is_directory=True
    )
    (tmp_path / "scripts").symlink_to(
        repository_root / "scripts", target_is_directory=True
    )
    (tmp_path / "data/raw/tables").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        load, "verify_against_manifest", lambda *_: calls.append("check-inputs")
    )
    for operation in ("create", "start", "stop", "close"):
        monkeypatch.setattr(
            PostgresContainer,
            operation,
            lambda *_, operation=operation: calls.append(operation),
        )
    monkeypatch.setattr(
        PostgresContainer,
        "restore_archive",
        lambda *_: pytest.fail("load must not restore"),
    )
    monkeypatch.setattr(
        load.PostgresWorker, "admin_sql", lambda _, sql: calls.append("load-sql")
    )

    def verify(*args, **kwargs):
        calls.append("verify")
        if verification_fails:
            raise RuntimeError("verification failed")

    monkeypatch.setattr(load, "verify", verify)
    monkeypatch.setattr(load, "archive_database", lambda *_: calls.append("archive"))

    if verification_fails:
        with pytest.raises(RuntimeError, match="verification failed"):
            load.load_verify_archive(tmp_path)
        assert calls == ["check-inputs", "create", "start", "load-sql", "verify"]
    else:
        load.load_verify_archive(tmp_path)
        assert calls == [
            "check-inputs",
            "create",
            "start",
            "load-sql",
            "verify",
            "stop",
            "archive",
            "close",
        ]


def test_load_requires_fetched_inputs_and_preserves_existing_archive(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError, match=r"run `uv run python -m scripts\.imdb\.fetch` first"
    ):
        load.load_verify_archive(tmp_path)
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        load.load_verify_archive(tmp_path)
    assert archive.read_bytes() == b"original"


@pytest.mark.parametrize("state", ["false 0", "true 0", "false 1"])
def test_archive_requires_clean_shutdown_and_writes_no_manifest(
    repository_root: Path, tmp_path: Path, monkeypatch, state: str
) -> None:
    profile = load_runtime_profile(repository_root, load.POOL_CONFIG)
    archive = tmp_path / "imdb.tar.gz"
    container = PostgresContainer(
        DatabaseFixture(repository_root, archive),
        "test-archive",
        profile,
        profile.workers[0],
    )
    container.container = "container"
    container.volume = "volume"
    container.image_id = "sha256:image"
    container.pgdata_relative_path = "18/docker"
    calls = []

    def command(arguments):
        calls.append(arguments)
        if arguments[:2] == ["docker", "inspect"]:
            return state
        assert "volume:/source:ro" in arguments
        assert "--network=none" in arguments
        assert arguments[-1] == "18/docker"
        (tmp_path / ".imdb.tar.gz.part").write_bytes(b"prepared archive")
        return ""

    monkeypatch.setattr(container, "command", command)
    if state != "false 0":
        with pytest.raises(RuntimeError, match="stopped cleanly"):
            load.archive_database(container, archive)
        assert not archive.exists()
        assert len(calls) == 1
    else:
        load.archive_database(container, archive)
        assert archive.read_bytes() == b"prepared archive"
        assert list(tmp_path.iterdir()) == [archive]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            load.archive_database(container, archive)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (None, None),
        ("missing", "missing=\\['title.csv'\\]"),
        ("unexpected", "unexpected=\\['extra.csv'\\]"),
        ("size", "Size mismatch"),
        ("checksum", "SHA-256 mismatch"),
    ],
    ids=["valid", "missing", "unexpected", "wrong-size", "wrong-checksum"],
)
def test_verify_against_manifest_checks_extracted_files(
    repository_root: Path, tmp_path: Path, change: str | None, error: str | None
) -> None:
    manifest = ImdbManifest.model_validate_json(
        (repository_root / "scripts/imdb/manifest.json").read_text()
    )
    contents = b"expected"
    members = {
        name: ImdbMember(bytes=len(contents), sha256=sha256_bytes(contents))
        for name in ("title.csv", "schematext.sql")
    }
    manifest = manifest.model_copy(
        update={"dataset": manifest.dataset.model_copy(update={"members": members})}
    )
    manifest_path = tmp_path / "scripts/imdb/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json())
    target = tmp_path / "data/raw/tables"
    target.mkdir(parents=True)
    for name in members:
        if change != "missing" or name != "title.csv":
            (target / name).write_bytes(contents)
    if change == "unexpected":
        (target / "extra.csv").write_bytes(contents)
    elif change == "size":
        (target / "title.csv").write_bytes(b"short")
    elif change == "checksum":
        (target / "title.csv").write_bytes(b"modified")

    if error is None:
        load.verify_against_manifest(tmp_path)
    else:
        with pytest.raises(RuntimeError, match=error):
            load.verify_against_manifest(tmp_path)


def test_verify_against_manifest_rejects_invalid_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "scripts/imdb/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")

    with pytest.raises(ValidationError, match="dataset"):
        load.verify_against_manifest(tmp_path)


def test_load_database_state_renders_sql_file(
    repository_root: Path, tmp_path: Path, monkeypatch, database_state
) -> None:
    query = Mock(return_value=json.dumps(database_state))
    monkeypatch.setattr(load, "admin_psql", query)
    monkeypatch.chdir(tmp_path)

    assert (
        load.load_database_state("container", ["title", "movie_info"]) == database_state
    )

    expected_sql = (
        (repository_root / "scripts/imdb/get_snapshot.sql")
        .read_text(encoding="utf-8")
        .format(
            table_array="'title', 'movie_info'",
            row_queries=(
                "SELECT 'title' AS table_name, count(*)::bigint AS row_count "
                "FROM public.title\nUNION ALL\n"
                "SELECT 'movie_info' AS table_name, count(*)::bigint AS row_count "
                "FROM public.movie_info"
            ),
        )
    )
    query.assert_called_once_with("container", expected_sql)


@pytest.fixture
def database_state(repository_root: Path):
    metadata = json.loads((repository_root / "scripts/imdb/manifest.json").read_text())
    rows = {
        item["table"]: item["rows"]
        for item in metadata["dataset"]["members"].values()
        if "table" in item
    }
    return {
        "identity": metadata["database"],
        "table_names": sorted(rows),
        "table_rows": rows,
        "columns": [{"table": "title", "column": "id"}],
        "constraints": [{"table": "title", "definition": "primary key"}],
        "indexes": [
            {
                "name": f"index-{index}",
                "primary": index < 21,
                "valid": True,
                "ready": True,
            }
            for index in range(44)
        ],
        "statistics": [{"table": name, "column": "id"} for name in rows],
        "relations": [{"table": name, "frozen_xid_age": 0} for name in rows],
    }


@pytest.mark.parametrize(
    ("changed", "error"),
    [
        (None, None),
        ("table_names", "IMDb table-set mismatch"),
        ("table_rows", "IMDb row-count mismatch"),
        ("identity", "database identity mismatch"),
        ("indexes", "unexpected total index count"),
        ("unready_index", "invalid or unready indexes"),
        ("statistics", "planner statistics missing"),
        ("relations", "not freshly frozen"),
    ],
)
def test_verify_loaded_database(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
    database_state,
    changed: str | None,
    error: str | None,
) -> None:
    state = copy.deepcopy(database_state)
    outputs = {"1a.sql": {"csv": "result\n"}}
    if changed == "table_names":
        state["table_names"].pop()
    elif changed == "table_rows":
        state["table_rows"]["title"] += 1
    elif changed == "identity":
        state["identity"]["encoding"] = "LATIN1"
    elif changed == "indexes":
        state["indexes"].pop()
    elif changed == "unready_index":
        state["indexes"][0]["ready"] = False
    elif changed == "statistics":
        state["statistics"].pop()
    elif changed == "relations":
        state["relations"][0]["frozen_xid_age"] = load.MAX_FRESHLY_FROZEN_XID_AGE + 1
    monkeypatch.setattr(load, "run", lambda *_: "")
    monkeypatch.setattr(load, "load_database_state", lambda *_: state)
    monkeypatch.setattr(load, "representative_query_outputs", lambda *_: outputs)
    report = tmp_path / "loaded.json"

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            load.verify("container", report, repository=repository_root)
        assert not report.exists()
    else:
        load.verify("container", report, repository=repository_root)
        result = json.loads(report.read_text())
        assert result["phase"] == "load"
        assert result["database"] == state
        assert result["representative_query_outputs"] == outputs
        assert result["fingerprints"] == {
            **{
                name: load.fingerprint(state[name])
                for name in (
                    "table_names",
                    "table_rows",
                    "columns",
                    "constraints",
                    "indexes",
                    "statistics",
                )
            },
            "representative_query_outputs": load.fingerprint(outputs),
        }


def test_load_rejects_invalid_inputs_before_creating_a_container(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "data/raw/tables").mkdir(parents=True)
    monkeypatch.setattr(
        load,
        "verify_against_manifest",
        Mock(side_effect=RuntimeError("invalid inputs")),
    )
    container = Mock(side_effect=AssertionError("invalid inputs must not be loaded"))
    monkeypatch.setattr(load, "PostgresContainer", container)

    with pytest.raises(RuntimeError, match="invalid inputs"):
        load.load_verify_archive(tmp_path)
    container.assert_not_called()


def test_load_sql_finalizes_exactly_the_expected_tables(repository_root: Path) -> None:
    metadata = json.loads((repository_root / "scripts/imdb/manifest.json").read_text())
    sql = (repository_root / "scripts/imdb/load.sql").read_text()
    vacuum = sql.split("VACUUM (FREEZE, ANALYZE)", 1)[1].split(";", 1)[0]
    tables = [name.strip().removeprefix("public.") for name in vacuum.split(",")]
    assert tables == metadata["load"]["table_order"]
    assert sql.index("create index ") < sql.index("VACUUM") < sql.index("CHECKPOINT;")
