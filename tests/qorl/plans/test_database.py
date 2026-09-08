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
from pydantic import JsonValue, TypeAdapter

from qorl.agent.interface import AgentInterface
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.measure.schemas import Candidate
from qorl.measure.validation import PlanValidationEvaluator
from qorl.plans.verify import matching_join, memoized_inner
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.inspection import column_statistics, inspect_relation
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


def test_live_inspection_contract(
    database: PostgresClient, record_property: Callable[[str, str], None]
) -> None:
    metadata = inspect_relation(database.admin_sql, "audit_a")
    stats = column_statistics(
        database.admin_sql, "audit_a", ["id", "k", "payload", "absent"]
    )
    record_property("relation", metadata.model_dump_json())
    record_property("statistics", stats.model_dump_json())
    assert metadata.exists and metadata.estimated_rows == 20000
    assert metadata.table_bytes is not None and metadata.table_bytes > 0
    assert {item.name for item in metadata.indexes} == {
        "audit_a_id_idx",
        "audit_a_k_idx",
    }
    assert stats.columns[0].n_distinct == -1
    original_json = database.admin_sql(
        "SELECT to_jsonb(histogram_bounds) FROM pg_stats WHERE schemaname='public' AND tablename='audit_a' AND attname='id' AND NOT inherited;"
    )
    original = TypeAdapter(list[JsonValue]).validate_json(original_json)
    record_property("original_id_histogram", original_json)
    histogram = stats.columns[0]
    assert histogram.histogram_bounds is not None
    assert (
        histogram.histogram_bounds[0] == 1 and histogram.histogram_bounds[-1] == 20000
    )
    assert len(histogram.histogram_bounds) == 16
    assert histogram.histogram_bound_positions[0] == 0
    assert histogram.histogram_bound_positions[-1] == len(original) - 1
    assert histogram.histogram_bounds == [
        original[index] for index in histogram.histogram_bound_positions
    ]
    assert histogram.histogram_bound_count == len(original)
    assert histogram.omitted_histogram_bounds == len(original) - 16
    assert stats.columns[1].most_common_values is not None
    assert all(isinstance(value, int) for value in stats.columns[1].most_common_values)
    assert stats.columns[1].most_common_frequencies is not None
    assert stats.columns[1].omitted_common_values > 0
    assert stats.columns[2].most_common_values == ["a" * 20]
    assert stats.columns[3].status == "missing_column"
    assert not inspect_relation(database.admin_sql, "absent").exists


@pytest.mark.parametrize("alias_count", [3, 40])
def test_live_small_and_many_alias_tool_views(
    database: PostgresClient,
    record_property: Callable[[str, str], None],
    alias_count: int,
) -> None:
    relations = [Relation(alias=f"a{i}", table="audit_a") for i in range(alias_count)]
    edges = (
        ["a0:audit_a.id=a1:audit_a.id", "a1:audit_a.id=a2:audit_a.id"]
        if alias_count == 3
        else []
    )
    task = Task(
        task_id=f"inspection-{alias_count}",
        template_id="inspection",
        sql_path="synthetic.sql",
        sql_sha256="synthetic",
        relations=relations,
        tables=["audit_a"],
        join_edges=edges,
        table_count=1,
        relation_count=alias_count,
        join_predicate_count=len(edges),
    )
    sql = " UNION ALL ".join(
        f"SELECT id FROM audit_a {relation.alias} WHERE id < 3"
        for relation in relations
    )
    if alias_count == 3:
        sql = "SELECT a0.id FROM audit_a a0 JOIN audit_a a1 ON a0.id=a1.id JOIN audit_a a2 ON a1.id=a2.id WHERE a0.id < 3"
    run = PlanValidationEvaluator(
        database,
        Query(sql),
        task,
        default_timeout_ms=QUERY_TIMEOUT_MS,
        max_candidates=1,
    )
    run.start()
    interface = AgentInterface.from_evaluator(run, 64)
    runtime = AgentEnvironment(run)
    record_property("observation", interface.observation.model_dump_json())
    record_property(
        "messages",
        json.dumps(
            [
                message.model_dump(mode="json", exclude_none=True)
                for message in interface.initial_messages()
            ]
        ),
    )
    record_property(
        "tools", json.dumps([tool.model_dump(mode="json") for tool in interface.tools])
    )
    requests: list[tuple[str, dict[str, JsonValue]]] = [
        ("inspect_relation", {"relation": "a0"}),
        ("get_column_stats", {"relation": "a0", "columns": ["id", "k"]}),
        ("get_plan", {"candidate_id": "default"}),
        ("evaluate_candidate", {"action": {"version": 1}}),
        ("finish", {}),
    ]
    for name, arguments in requests:
        result, _ = runtime.execute(name, arguments)
        assert "error" not in result
        record_property(name, json.dumps(result))
    kept = PlanValidationEvaluator(
        database,
        Query(sql),
        task,
        default_timeout_ms=QUERY_TIMEOUT_MS,
        max_candidates=1,
    )
    kept.start()
    result, finished = AgentEnvironment(kept).execute("keep_default", {})
    assert finished and result["status"] == "kept_default"
    record_property("keep_default", json.dumps(result))
    assert interface.observation.default_plan.nodes[0].leaf_aliases == sorted(
        relation.alias for relation in relations
    )


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
