"""Typed contents of the IMDb source and loading manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImdbRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImdbFile(ImdbRecord):
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImdbArchive(ImdbFile):
    filename: str = Field(min_length=1)


class ImdbMember(ImdbFile):
    table: str | None = None
    rows: int | None = Field(default=None, ge=0)


class ImdbCopyFormat(ImdbRecord):
    format: Literal["csv"]
    header: bool
    delimiter: str
    quote: str
    escape: str
    null: str


class ImdbDataset(ImdbRecord):
    source_url: str
    archive: ImdbArchive
    copy_format: ImdbCopyFormat
    members: dict[str, ImdbMember]


class ImdbDatabase(ImdbRecord):
    image_reference: str
    server_version_num: str
    encoding: str
    collation: str
    ctype: str
    expected_table_count: int = Field(ge=1)
    expected_primary_key_count: int = Field(ge=0)
    expected_secondary_index_count: int = Field(ge=0)
    expected_total_index_count: int = Field(ge=0)


class ImdbFinalization(ImdbRecord):
    vacuum: str
    statistics_target: int = Field(ge=0)
    checkpoint_after_vacuum: bool


class ImdbLoad(ImdbRecord):
    table_order: list[str]
    finalization: ImdbFinalization


class ImdbManifest(ImdbRecord):
    schema_version: int = Field(ge=1)
    fixture_id: Literal["imdb"]
    description: str
    dataset: ImdbDataset
    database: ImdbDatabase
    load: ImdbLoad
