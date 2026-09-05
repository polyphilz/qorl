import json
from pathlib import Path
from typing import assert_type

import pytest
from pydantic import ValidationError

from scripts.imdb.schemas import (
    ImdbArchive,
    ImdbCopyFormat,
    ImdbDatabase,
    ImdbDataset,
    ImdbFinalization,
    ImdbLoad,
    ImdbManifest,
    ImdbMember,
)


def test_manifest_models_cover_the_complete_document(repository_root: Path) -> None:
    text = (repository_root / "scripts/imdb/manifest.json").read_text()
    manifest = ImdbManifest.model_validate_json(text)

    assert_type(manifest.dataset, ImdbDataset)
    assert_type(manifest.dataset.archive, ImdbArchive)
    assert_type(manifest.dataset.archive.filename, str)
    assert_type(manifest.dataset.archive.bytes, int)
    assert_type(manifest.dataset.copy_format, ImdbCopyFormat)
    assert_type(manifest.dataset.members, dict[str, ImdbMember])
    assert_type(manifest.database, ImdbDatabase)
    assert_type(manifest.load, ImdbLoad)
    assert_type(manifest.load.finalization, ImdbFinalization)
    assert manifest.model_dump(exclude_unset=True) == json.loads(text)
    assert manifest.dataset.members["title.csv"].table == "title"
    assert manifest.dataset.members["title.csv"].rows is not None
    assert manifest.dataset.members["schematext.sql"].table is None
    assert manifest.dataset.members["schematext.sql"].rows is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "1"),
        (("fixture_id",), "job"),
        (("dataset", "archive", "filename"), 123),
        (("dataset", "archive", "bytes"), "123"),
        (("dataset", "archive", "bytes"), True),
        (("dataset", "archive", "bytes"), -1),
        (("dataset", "archive", "sha256"), "not-a-checksum"),
        (("dataset", "archive", "unexpected"), 1),
        (("dataset", "copy_format", "header"), "false"),
        (("dataset", "members", "title.csv", "rows"), "123"),
        (("database", "expected_table_count"), "21"),
        (("load", "table_order"), "title"),
        (("load", "finalization", "checkpoint_after_vacuum"), "true"),
    ],
    ids=[
        "version-string",
        "wrong-fixture",
        "filename-number",
        "size-string",
        "size-boolean",
        "size-negative",
        "checksum-malformed",
        "unknown-field",
        "header-string",
        "row-count-string",
        "table-count-string",
        "table-order-string",
        "checkpoint-string",
    ],
)
def test_invalid_manifest_fields_are_rejected(
    repository_root: Path, path: tuple[str, ...], value: str | int | bool
) -> None:
    document = json.loads((repository_root / "scripts/imdb/manifest.json").read_text())
    parent = document
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value

    with pytest.raises(ValidationError) as error:
        ImdbManifest.model_validate_json(json.dumps(document))
    assert error.value.errors()[0]["loc"] == path


@pytest.mark.parametrize("section", ["dataset", "database", "load"])
def test_required_manifest_sections_cannot_be_missing(
    repository_root: Path, section: str
) -> None:
    document = json.loads((repository_root / "scripts/imdb/manifest.json").read_text())
    del document[section]

    with pytest.raises(ValidationError) as error:
        ImdbManifest.model_validate_json(json.dumps(document))
    assert error.value.errors()[0]["loc"] == (section,)
