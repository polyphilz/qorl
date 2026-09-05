import json
from pathlib import Path

import pytest

from qorl.workload.ceb import SHORT_BINUNICODE_OPCODE
from scripts.benchmarks import extract_sql_from_qreps as extractor


def test_build_and_check_without_recovery_report(
    repository_root: Path, tmp_path: Path, monkeypatch
) -> None:
    manifest = json.loads(
        (repository_root / "benchmarks/ceb/manifest.json").read_text()
    )
    queries = manifest["queries"]
    queries.update(count=1, templates={"1a": 1})
    queries["subsets"]["unique_plans"].update(count=1, templates={"1a": 1})
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps(manifest))
    monkeypatch.setattr(extractor, "REPOSITORY_ROOT", tmp_path)
    sql = b"SELECT MIN(t.id) FROM title AS t;\n"
    qrep = bytes([SHORT_BINUNICODE_OPCODE, len(sql)]) + sql
    full = tmp_path / "full"
    unique = tmp_path / "unique"
    for source in (full, unique):
        (source / "1a").mkdir(parents=True)
        (source / "1a/query.pkl").write_bytes(qrep)
    output = tmp_path / "ceb"

    extractor.build(config, full, unique, output)
    extractor.check(config, output)

    query = output / "queries/1a/query.sql"
    assert query.read_bytes() == sql
    assert not (output / "provenance/recovery.json").exists()
    query.write_bytes(b"SELECT 1;\n")
    with pytest.raises(RuntimeError, match="checked-in SQL differs"):
        extractor.check(config, output)
