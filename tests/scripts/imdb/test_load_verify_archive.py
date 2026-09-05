import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from qorl.postgres.client import PostgresClient
from qorl.util.hashing import sha256_bytes
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import ContainerPool
from scripts.imdb import load_verify_archive as load
from scripts.imdb.schemas import (
    DatabaseColumn,
    DatabaseConstraint,
    DatabaseIdentity,
    DatabaseIndex,
    DatabaseRelation,
    DatabaseSnapshot,
    DatabaseStatistic,
    ImdbManifest,
    ImdbMember,
    LoadVerificationReport,
    RepresentativeQueryOutput,
)


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


def test_module_import(repository_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-E", "-c", "import scripts.imdb.load_verify_archive"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


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
    monkeypatch.setattr(load, "REPOSITORY_ROOT", tmp_path)
    input_manifests: list[ImdbManifest] = []

    def verify_inputs(repository: Path, manifest: ImdbManifest) -> None:
        assert repository == tmp_path
        input_manifests.append(manifest)
        calls.append("check-inputs")

    monkeypatch.setattr(load, "verify_input_csvs_against_manifest", verify_inputs)
    for operation in ("create", "start", "stop", "close"):
        monkeypatch.setattr(
            ContainerPool,
            operation,
            lambda *_, operation=operation: calls.append(operation),
        )
    monkeypatch.setattr(
        ContainerPool,
        "restore",
        lambda *_: pytest.fail("load must not restore"),
    )
    monkeypatch.setattr(
        load.PostgresClient, "admin_sql", lambda _, sql: calls.append("load-sql")
    )

    def verify(*args, **kwargs):
        assert kwargs["manifest"] is input_manifests[0]
        calls.append("verify")
        if verification_fails:
            raise RuntimeError("verification failed")

    monkeypatch.setattr(load, "verify_load", verify)
    monkeypatch.setattr(load, "archive_database", lambda *_: calls.append("archive"))

    if verification_fails:
        with pytest.raises(RuntimeError, match="verification failed"):
            load.main()
        assert calls == ["check-inputs", "create", "start", "load-sql", "verify"]
    else:
        load.main()
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
    monkeypatch,
) -> None:
    monkeypatch.setattr(load, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(
        RuntimeError, match=r"run `uv run python -m scripts\.imdb\.fetch` first"
    ):
        load.main()
    archive = tmp_path / "data/imdb.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        load.main()
    assert archive.read_bytes() == b"original"


@pytest.mark.parametrize("state", ["false 0", "true 0", "false 1"])
def test_archive_requires_clean_shutdown_and_writes_no_manifest(
    repository_root: Path, tmp_path: Path, monkeypatch, state: str
) -> None:
    profile = load_pool_config(repository_root, load.POOL_CONFIG)
    archive = tmp_path / "imdb.tar.gz"
    pool = ContainerPool(
        repository_root,
        "test-archive",
        profile,
        load.PostgresConfig.load(repository_root, load.POSTGRES_CONFIG),
    )
    slot = pool.workers[0]
    slot.container_id = "container"
    slot.volume = "volume"
    slot.image_id = "sha256:image"
    slot.pgdata_relative_path = "18/docker"
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

    monkeypatch.setattr(pool, "command", command)
    if state != "false 0":
        with pytest.raises(RuntimeError, match="stopped cleanly"):
            load.archive_database(pool, slot, archive)
        assert not archive.exists()
        assert len(calls) == 1
    else:
        load.archive_database(pool, slot, archive)
        assert archive.read_bytes() == b"prepared archive"
        assert list(tmp_path.iterdir()) == [archive]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            load.archive_database(pool, slot, archive)


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
def test_verify_input_csvs_against_manifest_checks_extracted_files(
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
        load.verify_input_csvs_against_manifest(tmp_path, manifest)
    else:
        with pytest.raises(RuntimeError, match=error):
            load.verify_input_csvs_against_manifest(tmp_path, manifest)


def test_load_rejects_invalid_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "scripts/imdb/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    (tmp_path / "data/raw/tables").mkdir(parents=True)
    monkeypatch.setattr(load, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(ValidationError, match="dataset"):
        load.main()


def test_load_database_snapshot_renders_sql_file(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch,
    database_snapshot: DatabaseSnapshot,
) -> None:
    query = Mock(return_value=database_snapshot.model_dump_json())
    client = PostgresClient(Mock())
    monkeypatch.setattr(client, "admin_sql", query)
    monkeypatch.chdir(tmp_path)

    assert (
        load.load_database_snapshot(client, ["title", "movie_info"])
        == database_snapshot
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
    query.assert_called_once_with(expected_sql)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("identity", "system_identifier", 123),
        ("columns", "ordinal", "1"),
        ("constraints", "validated", "true"),
        ("indexes", "valid", 1),
        ("statistics", "avg_width", "4"),
        ("statistics", "null_frac", True),
        ("relations", "relation_bytes", "8192"),
        ("table_rows", "title", "2528312"),
    ],
    ids=[
        "identity",
        "column",
        "constraint",
        "index",
        "statistics",
        "boolean-statistic",
        "relation",
        "row-count",
    ],
)
def test_load_database_snapshot_validates_nested_json(
    database_snapshot: DatabaseSnapshot,
    monkeypatch,
    section: str,
    field: str,
    value: str | int | bool,
) -> None:
    raw = database_snapshot.model_dump()
    row = raw[section]
    if isinstance(row, list):
        row = row[0]
    row[field] = value
    client = PostgresClient(Mock())
    monkeypatch.setattr(client, "admin_sql", Mock(return_value=json.dumps(raw)))
    with pytest.raises(ValidationError) as error:
        load.load_database_snapshot(client, ["title"])
    assert error.value.errors()[0]["loc"][0] == section


def test_snapshot_preserves_postgres_json_values(
    database_snapshot: DatabaseSnapshot,
) -> None:
    raw = database_snapshot.model_dump(mode="json")
    raw["constraints"] = None
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parsed = DatabaseSnapshot.model_validate_json(encoded)
    assert parsed.constraints is None
    assert parsed.columns is not None
    assert parsed.columns[0].default is None
    assert parsed.statistics is not None
    assert type(parsed.statistics[0].null_frac) is int
    assert type(parsed.statistics[0].correlation) is float
    assert (
        json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        == encoded
    )


def test_representative_query_outputs_records_exact_csv_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sql = "SELECT 'café' AS value;"
    (tmp_path / "1a.sql").write_text(sql, encoding="utf-8")
    csv = "value\ncafé\n"
    client = PostgresClient(Mock())
    query = Mock(return_value=csv)
    monkeypatch.setattr(client, "runner_sql", query)

    outputs = load.representative_query_outputs(client, tmp_path, ["1a.sql"])

    assert list(outputs) == ["1a.sql"]
    assert outputs["1a.sql"].csv == csv
    assert outputs["1a.sql"].bytes == len(csv.encode("utf-8"))
    assert outputs["1a.sql"].sha256 == sha256_bytes(csv.encode("utf-8"))
    query.assert_called_once_with(
        sql, csv=True, application_name="qorl-imdb-query-verifier"
    )


def test_section_checksum_is_canonical() -> None:
    expected = sha256_bytes(b'{"a":0,"b":"caf\xc3\xa9"}')
    assert load.section_checksum({"b": "café", "a": 0}) == expected
    assert load.section_checksum({"a": 0, "b": "café"}) == expected


@pytest.mark.parametrize("shape", ["record", "list", "mapping"])
def test_section_checksum_serializes_records_without_changing_hashes(
    shape: str,
) -> None:
    output = RepresentativeQueryOutput(
        csv="café\n",
        bytes=len("café\n".encode()),
        sha256=sha256_bytes("café\n".encode()),
    )
    if shape == "record":
        actual = load.section_checksum(output)
        plain = output.model_dump()
    elif shape == "list":
        actual = load.section_checksum([output])
        plain = [output.model_dump()]
    else:
        actual = load.section_checksum({"1a.sql": output})
        plain = {"1a.sql": output.model_dump()}
    expected = sha256_bytes(
        json.dumps(
            plain, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    assert actual == expected
    changed = output.model_copy(update={"csv": "different\n"})
    assert load.section_checksum(changed) != load.section_checksum(output)


@pytest.fixture
def imdb_manifest(repository_root: Path) -> ImdbManifest:
    return ImdbManifest.model_validate_json(
        (repository_root / "scripts/imdb/manifest.json").read_text()
    )


@pytest.fixture
def database_snapshot(imdb_manifest: ImdbManifest) -> DatabaseSnapshot:
    rows = {
        item.table: item.rows
        for item in imdb_manifest.dataset.members.values()
        if item.table is not None and item.rows is not None
    }
    database = imdb_manifest.database
    return DatabaseSnapshot(
        identity=DatabaseIdentity(
            server_version_num=database.server_version_num,
            database="imdb",
            encoding=database.encoding,
            collation=database.collation,
            ctype=database.ctype,
            system_identifier="123456789",
            pg_hint_plan_version="1.8.0",
        ),
        table_names=sorted(rows),
        table_rows=rows,
        columns=[
            DatabaseColumn(
                table="title",
                ordinal=1,
                column="id",
                data_type="integer",
                udt_name="int4",
                nullable="NO",
                default=None,
                character_maximum_length=None,
                numeric_precision=32,
                numeric_scale=0,
            )
        ],
        constraints=[
            DatabaseConstraint(
                table="title",
                name="title_pkey",
                type="p",
                definition="PRIMARY KEY (id)",
                validated=True,
            )
        ],
        indexes=[
            DatabaseIndex(
                table="title",
                name=f"index-{index}",
                definition="CREATE INDEX...",
                primary=index < 21,
                unique=index < 21,
                valid=True,
                ready=True,
            )
            for index in range(44)
        ],
        statistics=[
            DatabaseStatistic(
                table=name,
                column="id",
                inherited=False,
                null_frac=0,
                avg_width=4,
                n_distinct=-1,
                most_common_vals=None,
                most_common_freqs=None,
                histogram_bounds="{1,2,3}",
                correlation=1.0,
                most_common_elems=None,
                most_common_elem_freqs=None,
                elem_count_histogram=None,
            )
            for name in rows
        ],
        relations=[
            DatabaseRelation(
                table=name,
                relpages=1,
                reltuples=count,
                relallvisible=1,
                relfrozenxid="123",
                frozen_xid_age=0,
                relation_bytes=8192,
                total_relation_bytes=16384,
            )
            for name, count in rows.items()
        ],
    )


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
    database_snapshot: DatabaseSnapshot,
    imdb_manifest: ImdbManifest,
    changed: str | None,
    error: str | None,
) -> None:
    raw = database_snapshot.model_dump()
    outputs = {
        "1a.sql": RepresentativeQueryOutput(
            csv="result\n", bytes=len(b"result\n"), sha256=sha256_bytes(b"result\n")
        )
    }
    if changed == "table_names":
        raw["table_names"].pop()
    elif changed == "table_rows":
        raw["table_rows"]["title"] += 1
    elif changed == "identity":
        raw["identity"]["encoding"] = "LATIN1"
    elif changed == "indexes":
        raw["indexes"].pop()
    elif changed == "unready_index":
        raw["indexes"][0]["ready"] = False
    elif changed == "statistics":
        raw["statistics"].pop()
    elif changed == "relations":
        raw["relations"][0]["frozen_xid_age"] = load.MAX_FRESHLY_FROZEN_XID_AGE + 1
    state = DatabaseSnapshot.model_validate(raw)
    client = PostgresClient(Mock())
    monkeypatch.setattr(client, "execute", Mock())
    monkeypatch.setattr(load, "load_database_snapshot", lambda *_: state)
    query_results = Mock(return_value=outputs)
    monkeypatch.setattr(load, "representative_query_outputs", query_results)
    report = tmp_path / "loaded.json"

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            load.verify_load(
                client, report, manifest=imdb_manifest, repository=repository_root
            )
        assert not report.exists()
    else:
        load.verify_load(
            client, report, manifest=imdb_manifest, repository=repository_root
        )
        result = LoadVerificationReport.model_validate_json(report.read_text())
        assert result.schema_version == 2
        assert result.phase == "load"
        assert result.database == state
        assert result.representative_query_outputs == outputs
        assert result.checksums.model_dump() == {
            **{
                name: load.section_checksum(raw[name])
                for name in (
                    "table_names",
                    "table_rows",
                    "columns",
                    "constraints",
                    "indexes",
                    "statistics",
                )
            },
            "representative_query_outputs": load.section_checksum(
                {name: output.model_dump() for name, output in outputs.items()}
            ),
        }
        query_results.assert_called_once_with(
            client=client,
            source_dir=repository_root / "benchmarks/job/queries",
            query_names=["1a.sql", "17b.sql", "33c.sql"],
        )
        legacy = json.loads(report.read_text())
        legacy["schema_version"] = 1
        legacy["fingerprints"] = legacy.pop("checksums")
        assert (
            LoadVerificationReport.model_validate_json(json.dumps(legacy)).checksums
            == result.checksums
        )


def test_load_rejects_invalid_inputs_before_creating_a_container(
    repository_root: Path, tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "data/raw/tables").mkdir(parents=True)
    (tmp_path / "scripts").symlink_to(
        repository_root / "scripts", target_is_directory=True
    )
    monkeypatch.setattr(
        load,
        "verify_input_csvs_against_manifest",
        Mock(side_effect=RuntimeError("invalid inputs")),
    )
    monkeypatch.setattr(load, "REPOSITORY_ROOT", tmp_path)
    container = Mock(side_effect=AssertionError("invalid inputs must not be loaded"))
    monkeypatch.setattr(load, "ContainerPool", container)

    with pytest.raises(RuntimeError, match="invalid inputs"):
        load.main()
    container.assert_not_called()


def test_load_sql_finalizes_exactly_the_expected_tables(repository_root: Path) -> None:
    metadata = json.loads((repository_root / "scripts/imdb/manifest.json").read_text())
    sql = (repository_root / "scripts/imdb/load.sql").read_text()
    vacuum = sql.split("VACUUM (FREEZE, ANALYZE)", 1)[1].split(";", 1)[0]
    tables = [name.strip().removeprefix("public.") for name in vacuum.split(",")]
    assert tables == metadata["load"]["table_order"]
    assert sql.index("create index ") < sql.index("VACUUM") < sql.index("CHECKPOINT;")
