"""Opt-in live hint regressions; QORL_TEST_POSTGRES_IMAGE selects a prebuilt image."""

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import JsonValue

from qorl.measure.schemas import Candidate
from qorl.measure.validation import PlanValidationEvaluator
from qorl.plans.verify import matching_join, memoized_inner
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes, PostgresSettings
from qorl.taskset.schemas import Relation, Task

STARTUP_TIMEOUT_SECONDS = 60
COMMAND_TIMEOUT_SECONDS = 30
POLL_SECONDS = 1
QUERY_TIMEOUT_MS = 5_000
FIXTURE_CPUS = "2"
FIXTURE_MEMORY = "1g"
FIXTURE_SHM = "128m"
FIXTURE_TMPFS = "/var/lib/postgresql:rw,size=768m"
TEST_PASSWORD = "synthetic-fixture-only"
ARCHIVED_EXTENSION_VERSION = "1.8.0"
SQL = "SELECT count(*) FROM audit_a a JOIN audit_b b ON a.k=b.k JOIN audit_c c ON b.k=c.k WHERE a.id < 100;"
MEMOIZE_SQL = (
    "SELECT count(*) FROM audit_a a JOIN audit_b b ON a.k=b.k WHERE a.id < 10000;"
)
LEADING: dict[str, JsonValue] = {"left": {"left": "a", "right": "b"}, "right": "c"}


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        check=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="module")
def database(
    record_testsuite_property: Callable[[str, str], None],
) -> Iterator[PostgresClient]:
    """Own one network-isolated, temporary database; never mount the IMDb archive."""
    image = os.environ.get("QORL_TEST_POSTGRES_IMAGE")
    if not image:
        pytest.skip("live PostgreSQL regressions require QORL_TEST_POSTGRES_IMAGE")
    image_id = command(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
    ).stdout.strip()
    record_testsuite_property("postgres_image_id", image_id)
    for name in ("commit", "source-sha256"):
        value = command(
            [
                "docker",
                "image",
                "inspect",
                image_id,
                "--format",
                '{{index .Config.Labels "io.qorl.pg_hint_plan.' + name + '"}}',
            ]
        ).stdout.strip()
        assert value
        record_testsuite_property("pg_hint_plan_" + name, value)

    container = "qorl-plan-regression-" + uuid4().hex
    try:
        command(
            [
                "docker",
                "run",
                "--detach",
                "--pull=never",
                "--name",
                container,
                "--network=none",
                "--no-healthcheck",
                "--cpus",
                FIXTURE_CPUS,
                "--memory",
                FIXTURE_MEMORY,
                "--shm-size",
                FIXTURE_SHM,
                "--tmpfs",
                FIXTURE_TMPFS,
                "--env",
                "POSTGRES_USER=qorl_admin",
                "--env",
                "POSTGRES_DB=qorl",
                "--env",
                f"POSTGRES_PASSWORD={TEST_PASSWORD}",
                "--env",
                f"QORL_RUNNER_PASSWORD={TEST_PASSWORD}",
                image_id,
                "postgres",
                "-c",
                "shared_preload_libraries=pg_hint_plan",
                "-c",
                "geqo=off",
                "-c",
                "autovacuum=off",
                "-c",
                "max_parallel_workers_per_gather=0",
            ]
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "pg_isready",
                    "--host=127.0.0.1",
                    "--username=qorl_admin",
                    "--dbname=qorl",
                ],
                text=True,
                capture_output=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
            if ready.returncode == 0:
                break
            if time.monotonic() >= deadline:
                pytest.fail(
                    "database did not become ready: "
                    + command(["docker", "logs", container]).stdout
                )
            time.sleep(POLL_SECONDS)

        def execute(
            arguments: list[str], query: str
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["docker", "exec", "--interactive", container, *arguments],
                input=query,
                text=True,
                capture_output=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )

        postgres = PostgresClient(
            execute,
            PostgresConfig.load(
                Path("docker/postgres/configs/000-pgconf-default")
            ).agent_settings,
            PostgresIndexes(by_table={}),
        )
        postgres.admin_sql(
            Path(__file__).with_name("fixtures").joinpath("database.sql").read_text()
        )
        postgres.settings = PostgresSettings.model_validate_json(
            postgres.admin_sql(
                "SELECT json_object_agg(name, setting) FROM pg_settings WHERE name IN ("
                + ",".join("'" + name + "'" for name in PostgresSettings.model_fields)
                + ");"
            )
        )
        postgres.indexes = postgres.read_indexes()
        actual_id = command(
            ["docker", "inspect", container, "--format", "{{.Image}}"]
        ).stdout.strip()
        assert actual_id == image_id
        record_testsuite_property(
            "postgres_container_id",
            command(
                ["docker", "inspect", container, "--format", "{{.Id}}"]
            ).stdout.strip(),
        )
        record_testsuite_property(
            "pg_hint_plan_versions",
            command(
                ["docker", "exec", container, "cat", "/usr/share/qorl/versions.json"]
            ).stdout,
        )
        command(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                "cd /usr/lib/postgresql/18/lib && sha256sum --check /usr/share/qorl/pg_hint_plan.so.sha256",
            ]
        )
        record_testsuite_property(
            "pg_hint_plan_binary",
            command(
                [
                    "docker",
                    "exec",
                    container,
                    "sha256sum",
                    "/usr/lib/postgresql/18/lib/pg_hint_plan.so",
                ]
            ).stdout.strip(),
        )
        yield postgres
    finally:
        command(["docker", "rm", "--force", "--volumes", container])


@dataclass(frozen=True)
class Query:
    sql: str

    def load_sql(self, task: Task) -> str:
        return self.sql


def evaluator(
    database: PostgresClient, *, memoize: bool = False
) -> PlanValidationEvaluator[PostgresClient]:
    aliases = "ab" if memoize else "abc"
    relations = [Relation(alias=alias, table="audit_" + alias) for alias in aliases]
    edges = [
        f"{left}:audit_{left}.k={right}:audit_{right}.k"
        for left, right in pairwise(aliases)
    ]
    task = Task(
        task_id="synthetic-" + aliases,
        template_id="synthetic-" + aliases,
        sql_path="synthetic.sql",
        sql_sha256="synthetic fixture",
        relations=relations,
        tables=[relation.table for relation in relations],
        join_edges=edges,
        table_count=len(relations),
        relation_count=len(relations),
        join_predicate_count=len(edges),
    )
    run = PlanValidationEvaluator(
        database,
        Query(MEMOIZE_SQL if memoize else SQL),
        task,
        default_timeout_ms=QUERY_TIMEOUT_MS,
        max_candidates=1,
    )
    run.start()
    return run


def record(candidate: Candidate, record_property: Callable[[str, str], None]) -> None:
    """Persist action, compiled hint, full plan, diagnostics, and model feedback in JUnit."""
    record_property("candidate", candidate.model_dump_json())
    record_property("feedback", json.dumps(candidate.feedback()))


@pytest.mark.parametrize(
    "constraint",
    [
        {"force": "hash", "forbid": ["merge"]},
        {"forbid": ["hash", "merge"]},
        {"force": "merge"},
    ],
)
def test_join_compilation_to_live_candidate_feedback(
    database: PostgresClient,
    record_property: Callable[[str, str], None],
    constraint: dict[str, JsonValue],
) -> None:
    run = evaluator(database)
    candidate = run.evaluate(
        {
            "version": 1,
            "leading": LEADING,
            "joins": [{"relations": ["a", "b"], **constraint}],
        }
    )
    record(candidate, record_property)
    assert candidate.action_valid and candidate.constraints_satisfied, (
        candidate.errors_or_diagnostics
    )
    assert (
        candidate.pg_hint_plan is not None
        and candidate.pg_hint_plan["duplicate"] == "(none)"
    )
    assert database.explain_analyze_calls == 0


@pytest.mark.parametrize(
    "scan",
    [
        {
            "relation": "a",
            "force": "index",
            "indexes": ["audit_a_id_idx"],
            "forbid": ["seq"],
        },
        {"relation": "a", "force": "seq", "forbid": ["index"]},
        {"relation": "a", "force": "bitmap", "indexes": ["audit_a_id_idx"]},
        {
            "relation": "b",
            "force": "index_only",
            "indexes": ["audit_b_k_idx"],
            "forbid": ["index"],
        },
        {"relation": "b", "forbid": ["seq", "bitmap"]},
        {"relation": "a", "forbid": ["index_only"]},
        {"relation": "a", "forbid": ["index", "index_only"]},
    ],
)
def test_scan_constraints_preserve_indexes_and_physical_methods(
    database: PostgresClient,
    record_property: Callable[[str, str], None],
    scan: dict[str, JsonValue],
) -> None:
    candidate = evaluator(database).evaluate({"version": 1, "scans": [scan]})
    record(candidate, record_property)
    assert candidate.constraints_satisfied, candidate.errors_or_diagnostics


def test_index_only_fallback_is_rejected(
    database: PostgresClient, record_property: Callable[[str, str], None]
) -> None:
    candidate = evaluator(database).evaluate(
        {
            "version": 1,
            "scans": [
                {
                    "relation": "a",
                    "force": "index_only",
                    "indexes": ["audit_a_id_idx"],
                    "forbid": ["index"],
                }
            ],
        }
    )
    record(candidate, record_property)
    assert candidate.action_valid and not candidate.constraints_satisfied
    assert "scan a uses index, not index_only" in candidate.errors_or_diagnostics


def test_impossible_constraint_uses_no_candidate_sql(
    database: PostgresClient, record_property: Callable[[str, str], None]
) -> None:
    run = evaluator(database)
    before = database.explain_calls
    candidate = run.evaluate(
        {"version": 1, "scans": [{"relation": "a", "forbid": ["index"]}]}
    )
    record(candidate, record_property)
    assert not candidate.action_valid
    assert database.explain_calls == before


@pytest.mark.parametrize("leading", [False, True])
@pytest.mark.parametrize("join_method", [False, True])
@pytest.mark.parametrize("mode", ["force", "forbid"])
def test_memoization_composes_with_leading_and_join_method(
    database: PostgresClient,
    record_property: Callable[[str, str], None],
    leading: bool,
    join_method: bool,
    mode: str,
) -> None:
    run = evaluator(database, memoize=True)
    assert run.default is not None
    plan = run.default.plain_explain["Plan"]
    assert isinstance(plan, dict)
    baseline_join = matching_join(plan, ["a", "b"])
    assert baseline_join is not None and memoized_inner(baseline_join), (
        "fixture must make baseline memoization viable"
    )
    constraint: dict[str, JsonValue] = {"relations": ["a", "b"], "memoize": mode}
    if join_method:
        constraint["force"] = "nestloop"
    action: dict[str, JsonValue] = {"version": 1, "joins": [constraint]}
    if leading:
        action["leading"] = {"left": "a", "right": "b"}
    candidate = run.evaluate(action)
    record(candidate, record_property)
    assert candidate.constraints_satisfied, candidate.errors_or_diagnostics
    assert candidate.pg_hint_plan is not None
    hint = "Memoize(a b)" if mode == "force" else "NoMemoize(a b)"
    assert hint in candidate.pg_hint_plan["used"]
    assert candidate.pg_hint_plan["not_used"] == "(none)"


def test_old_extension_catalog_updates_to_installed_source(
    database: PostgresClient, record_property: Callable[[str, str], None]
) -> None:
    """Exercise a physical archive's retained catalog version on the new binary."""
    expected = database.admin_sql(
        "SELECT default_version FROM pg_available_extensions WHERE name = 'pg_hint_plan';"
    ).strip()
    version_sql = "SELECT extversion FROM pg_extension WHERE extname = 'pg_hint_plan';"
    contents_sql = (
        "SELECT count(*), sum(id), sum(k) FROM audit_a UNION ALL "
        "SELECT count(*), sum(id), sum(k) FROM audit_b UNION ALL "
        "SELECT count(*), sum(id), sum(k) FROM audit_c;"
    )
    before = database.admin_sql(contents_sql)
    database.admin_sql(
        "DROP EXTENSION pg_hint_plan; "
        f"CREATE EXTENSION pg_hint_plan VERSION '{ARCHIVED_EXTENSION_VERSION}';"
    )
    assert database.admin_sql(version_sql).strip() == ARCHIVED_EXTENSION_VERSION
    database.admin_sql("ALTER EXTENSION pg_hint_plan UPDATE;")
    after = database.admin_sql(version_sql).strip()
    record_property("extension_catalog_before", ARCHIVED_EXTENSION_VERSION)
    record_property("extension_catalog_after", after)
    assert after == expected
    assert database.admin_sql(contents_sql) == before
