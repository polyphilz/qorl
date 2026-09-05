"""IMDb source manifest, captured database state, and load verification records."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ImdbRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImdbFile(ImdbRecord):
    bytes: int = Field(ge=0)
    sha256: Sha256


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


class DatabaseIdentity(ImdbRecord):
    server_version_num: str
    database: str
    encoding: str
    collation: str
    ctype: str
    system_identifier: str
    pg_hint_plan_version: str | None


class DatabaseColumn(ImdbRecord):
    table: str
    ordinal: int
    column: str
    data_type: str
    udt_name: str
    nullable: Literal["YES", "NO"]
    default: str | None
    character_maximum_length: int | None
    numeric_precision: int | None
    numeric_scale: int | None


class DatabaseConstraint(ImdbRecord):
    table: str
    name: str
    type: str
    definition: str
    validated: bool


class DatabaseIndex(ImdbRecord):
    table: str
    name: str
    definition: str
    primary: bool
    unique: bool
    valid: bool
    ready: bool


class DatabaseStatistic(ImdbRecord):
    table: str
    column: str
    inherited: bool
    null_frac: int | float
    avg_width: int
    n_distinct: int | float
    most_common_vals: str | None
    most_common_freqs: str | None
    histogram_bounds: str | None
    correlation: int | float | None
    most_common_elems: str | None
    most_common_elem_freqs: str | None
    elem_count_histogram: str | None


class DatabaseRelation(ImdbRecord):
    table: str
    relpages: int
    reltuples: int | float
    relallvisible: int
    relfrozenxid: str
    frozen_xid_age: int
    relation_bytes: int
    total_relation_bytes: int


class DatabaseSnapshot(ImdbRecord):
    identity: DatabaseIdentity
    table_names: list[str] | None
    table_rows: dict[str, int]
    columns: list[DatabaseColumn] | None
    constraints: list[DatabaseConstraint] | None
    indexes: list[DatabaseIndex] | None
    statistics: list[DatabaseStatistic] | None
    relations: list[DatabaseRelation] | None


class RepresentativeQueryOutput(ImdbFile):
    csv: str


class DatabaseChecksums(ImdbRecord):
    table_names: Sha256
    table_rows: Sha256
    columns: Sha256
    constraints: Sha256
    indexes: Sha256
    statistics: Sha256
    representative_query_outputs: Sha256


class LoadVerificationReport(ImdbRecord):
    schema_version: Literal[1, 2] = 2
    fixture_id: Literal["imdb"]
    phase: Literal["load"] = "load"
    captured_at_utc: str
    source_manifest_sha256: Sha256
    database: DatabaseSnapshot
    representative_query_outputs: dict[str, RepresentativeQueryOutput]
    checksums: DatabaseChecksums = Field(
        validation_alias=AliasChoices("checksums", "fingerprints")
    )
