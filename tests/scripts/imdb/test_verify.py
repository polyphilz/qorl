import copy
import json
from pathlib import Path

import pytest

from scripts.imdb import verify


@pytest.fixture
def database_state(repository_root: Path):
    metadata = json.loads(
        (repository_root / "scripts/imdb/imdb-metadata.json").read_text()
    )
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
    "changed",
    [
        None,
        "table_names",
        "table_rows",
        "columns",
        "constraints",
        "indexes",
        "statistics",
        "representative_query_outputs",
    ],
)
def test_restore_compares_every_recorded_fingerprint(
    repository_root: Path, tmp_path: Path, monkeypatch, database_state, changed
) -> None:
    state = copy.deepcopy(database_state)
    outputs = {"1a.sql": {"csv": "result\n"}}
    monkeypatch.setattr(verify, "run", lambda *_: "")
    monkeypatch.setattr(verify, "load_database_state", lambda *_: state)
    monkeypatch.setattr(verify, "representative_query_outputs", lambda *_: outputs)
    reference = tmp_path / "loaded.json"
    report = tmp_path / "restored.json"
    verify.verify("container", reference, repository=repository_root)

    if changed == "table_names":
        state[changed].pop()
    elif changed == "table_rows":
        state[changed]["title"] += 1
    elif changed == "representative_query_outputs":
        outputs["1a.sql"]["csv"] = "different result\n"
    elif changed is not None:
        state[changed][0]["changed"] = True

    if changed is None:
        verify.verify(
            "restored", report, repository=repository_root, compare_to=reference
        )
        result = json.loads(report.read_text())
        assert result["comparison"] == "passed"
        assert (
            result["fingerprints"] == json.loads(reference.read_text())["fingerprints"]
        )
    else:
        with pytest.raises(RuntimeError, match="mismatch"):
            verify.verify(
                "restored", report, repository=repository_root, compare_to=reference
            )
        assert not report.exists()


def test_load_sql_finalizes_exactly_the_expected_tables(repository_root: Path) -> None:
    metadata = json.loads(
        (repository_root / "scripts/imdb/imdb-metadata.json").read_text()
    )
    sql = (repository_root / "scripts/imdb/load.sql").read_text()
    vacuum = sql.split("VACUUM (FREEZE, ANALYZE)", 1)[1].split(";", 1)[0]
    tables = [name.strip().removeprefix("public.") for name in vacuum.split(",")]
    assert tables == metadata["load"]["table_order"]
    assert sql.index("fkindexes.sql") < sql.index("VACUUM") < sql.index("CHECKPOINT;")
