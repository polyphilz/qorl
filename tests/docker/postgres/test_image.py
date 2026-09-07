"""Source pins, build-stage scope, and active image references agree."""

import re
from pathlib import Path

from pydantic import BaseModel

from qorl.postgres.config import PostgresConfig
from scripts.imdb.schemas import ImdbManifest


class ExtensionSource(BaseModel):
    version: str
    branch: str
    commit: str
    source_archive_url: str
    source_archive_sha256: str


class Versions(BaseModel):
    pg_hint_plan: ExtensionSource


def test_source_pin_is_used_by_the_builder_and_runtime(repository_root: Path) -> None:
    directory = repository_root / "docker/postgres"
    source = Versions.model_validate_json(
        (directory / "versions.json").read_text()
    ).pg_hint_plan
    dockerfile = (directory / "Dockerfile").read_text()
    arguments = dict(re.findall(r"^ARG (\w+)=(.+)$", dockerfile, re.MULTILINE))
    assert arguments["PG_HINT_PLAN_COMMIT"] == source.commit
    assert arguments["PG_HINT_PLAN_VERSION"] == source.version
    assert arguments["PG_HINT_PLAN_SOURCE_SHA256"] == source.source_archive_sha256
    assert source.branch == "PG18"
    assert (
        source.source_archive_url
        == f"https://codeload.github.com/ossc-db/pg_hint_plan/tar.gz/{source.commit}"
    )
    assert "PG_HINT_PLAN_TAG" not in dockerfile
    assert 'io.qorl.pg_hint_plan.commit="${PG_HINT_PLAN_COMMIT}"' in dockerfile
    assert (
        'io.qorl.pg_hint_plan.source-sha256="${PG_HINT_PLAN_SOURCE_SHA256}"'
        in dockerfile
    )
    _, builder, runtime = dockerfile.split("FROM ${POSTGRES_BASE_IMAGE}")
    for stage in (builder, runtime):
        assert "ARG PG_HINT_PLAN_COMMIT\n" in stage
        assert "ARG PG_HINT_PLAN_SOURCE_SHA256\n" in stage
    assert (
        '"https://codeload.github.com/ossc-db/pg_hint_plan/tar.gz/${PG_HINT_PLAN_COMMIT}"'
        in builder
    )
    assert builder.index("sha256sum --check") < builder.index("make --directory source")
    assert "sha256sum --check /usr/share/qorl/pg_hint_plan.so.sha256" in runtime


def test_compose_and_fixture_loader_select_the_source_identified_image(
    repository_root: Path,
) -> None:
    compose = (repository_root / "compose.yaml").read_text()
    match = re.search(r"^    image: (\S+)$", compose, re.MULTILINE)
    assert match is not None
    image = match.group(1)
    source = Versions.model_validate_json(
        (repository_root / "docker/postgres/versions.json").read_text()
    ).pg_hint_plan
    revision = image.rsplit("-", 1)[-1]
    assert source.commit.startswith(revision)
    assert image.startswith(f"qorl-postgres:18.6-pg_hint_plan-{source.version}-")
    manifest = ImdbManifest.model_validate_json(
        (repository_root / "scripts/imdb/manifest.json").read_text()
    )
    assert manifest.database.image_reference == image


def test_config_expects_the_installed_extension_version(repository_root: Path) -> None:
    source = Versions.model_validate_json(
        (repository_root / "docker/postgres/versions.json").read_text()
    ).pg_hint_plan
    for directory in (repository_root / "docker/postgres/configs").iterdir():
        if directory.is_dir():
            config = PostgresConfig.load(directory)
            assert config.expected.postgresql.extension_version == source.version
