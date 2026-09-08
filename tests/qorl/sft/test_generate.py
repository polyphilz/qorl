"""Hosted translation and shared agent/evaluators with scripted external boundaries."""

import json
from pathlib import Path
from threading import Barrier
from urllib.parse import unquote, urlparse

import pytest
from jsonschema import validate
from renderers.configs import Qwen35RendererConfig
from tests.qorl.evaluation.conftest import EvaluationActivity
from tests.qorl.evaluation.conftest import activity as activity
from tests.qorl.sft.test_dataset import experiment as experiment
from tests.qorl.sft.test_dataset import student_model as student_model
from tests.qorl.sft.test_dataset import tokenizer as tokenizer
from tests.qorl.sft.test_dataset import tools as tools

from qorl.evaluation import evaluate
from qorl.experiment import run
from qorl.experiment.schemas import SftExperimentConfig
from qorl.measure.schemas import RunStatus
from qorl.model.client import HttpTransport
from qorl.model.exceptions import TransientModelError
from qorl.model.schemas import (
    AstraInferenceSettings,
    JsonObject,
    ModelPreset,
    ReasoningEffort,
)
from qorl.sft import dataset, generate
from qorl.sft.schemas import (
    Conversation,
    GenerationSettings,
    ImportedGenerationSeeds,
    RenderedConversation,
)
from qorl.taskset.schemas import TaskRole, TaskSelection
from qorl.util.hashing import sha256_file

HTTP_REQUEST = HttpTransport.request


@pytest.fixture
def configured(
    experiment: Path,
    activity: EvaluationActivity,
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]]:
    import tomllib

    inputs = run.load_inputs(experiment)
    assert isinstance(inputs.config, SftExperimentConfig)
    preset = ModelPreset.model_validate(
        tomllib.loads(
            (
                repository_root / "configs/defaults/models/000-gpt-6-astra.toml"
            ).read_text()
        )
    )
    generation = GenerationSettings(
        model=preset.model.model_copy(update={"max_concurrent_requests": 2}),
        inference=AstraInferenceSettings(
            max_tokens=1024, reasoning_effort=ReasoningEffort.MEDIUM
        ),
        generations_per_task=1,
    )
    config = inputs.config.model_copy(
        update={
            "data": inputs.config.data.model_copy(
                update={
                    "dataset_from": None,
                    "generation": generation,
                    "imported_generation_seeds": None,
                }
            ),
            "model": inputs.config.model.model_copy(update={"context_length": 32768}),
            "training": inputs.config.training.model_copy(
                update={"renderer": Qwen35RendererConfig()}
            ),
            "agent": inputs.config.agent.model_copy(
                update={"candidate_attempts": 3, "maximum_model_turns": 8}
            ),
        }
    )
    monkeypatch.setattr(generate, "start_pool", evaluate.start_pool)
    monkeypatch.setattr(generate, "capture_environment", evaluate.capture_environment)
    return SftExperimentConfig.model_validate_json(
        config.model_dump_json()
    ), inputs.selections


def attempts(output: Path) -> list[generate.GenerationAttempt]:
    return [
        generate.GenerationAttempt.model_validate_json(path.read_bytes())
        for path in sorted((output / "attempts").glob("*.json"))
    ]


@pytest.mark.parametrize("plan_only", [False, True])
def test_generate_prepare_and_reuse_original_evidence(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    plan_only: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    assert config.data.generation is not None
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": config.data.generation.model_copy(
                        update={"plan_only": plan_only}
                    )
                }
            )
        }
    )
    # Two concurrent conversations each make strictly sequential requests.
    activity.barrier = Barrier(2)
    output = tmp_path / "run"
    report = dataset.prepare_dataset(config, selections, output / "dataset")
    raw = attempts(output / "generation")
    assert len(raw) == 2
    assert all(item.status == RunStatus.COMPLETED for item in raw)
    assert all((item.plan_only is not None) == plan_only for item in raw)
    assert all((item.rollout is not None) != plan_only for item in raw)
    assert len(activity.pools) == len(activity.closed) == 1
    assert not activity.claimed_workers
    assert any("ANALYZE" in query for query in activity.queries) != plan_only
    assert report.training.accepted_requests == report.validation.accepted_requests == 3
    record = Conversation.model_validate_json(
        (output / "dataset/training.jsonl").read_bytes()
    )
    uri = record.metadata["generation_attempt"]
    assert isinstance(uri, str)
    original = Path(unquote(urlparse(uri).path))
    saved = generate.GenerationAttempt.model_validate_json(original.read_bytes())
    assert saved.attempt_id == record.conversation_id
    assert record.metadata["generation_identity_sha256"] == sha256_file(
        output / "generation/identity.json"
    )
    assert saved.trace is not None
    for request, actual in zip(
        record.requests, saved.trace.model_requests, strict=True
    ):
        assert record.messages[: request.assistant_message_index] == actual.messages
        assert request.tools == actual.tools
    rendered = RenderedConversation.model_validate_json(
        (output / "dataset/training/rendered/000000.json").read_bytes()
    )
    assert len(rendered.requests) == len(record.requests)
    before = len(activity.requests), len(activity.queries)
    assert (
        dataset.prepare_dataset(config, selections, output / "dataset", resume=True)
        == report
    )
    reuse = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": None,
                    "dataset_from": output / "dataset",
                    "imported_generation_seeds": ImportedGenerationSeeds(
                        training=config.experiment.seed,
                        validation=config.experiment.seed,
                    ),
                }
            )
        }
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset-from must not generate or execute SQL")

    monkeypatch.setattr(generate, "generate_dataset", forbidden)
    dataset.prepare_dataset(reuse, selections, tmp_path / "reused")
    assert (len(activity.requests), len(activity.queries)) == before
    assert (tmp_path / "reused/training.jsonl").read_bytes() == (
        output / "dataset/training.jsonl"
    ).read_bytes()
    assert original.exists()


@pytest.mark.parametrize(
    "mode,saved_count",
    [
        ("keep", 1),
        ("duplicate", 1),
        ("invalid", 0),
        ("candidate_timeout", 1),
        ("provider_failure", 0),
        ("provider_failure_after_candidate", 1),
    ],
)
def test_failures_and_exclusions_retain_attempts_without_replacement(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    mode: str,
    saved_count: int,
) -> None:
    config, selections = configured
    activity.mode = mode
    output = tmp_path / "generation"
    artifact = generate.generate_dataset(config, selections, output)
    records = attempts(output)
    assert len(records) == 2
    report = generate.GenerationReport.model_validate_json(
        (output / "report.json").read_bytes()
    )
    for split in (report.training, report.validation):
        assert (
            split.requested_attempts
            == split.completed_attempts + split.failed_attempts
            == 1
        )
        assert split.saved_conversations == saved_count
        assert split.saved_conversations + len(split.conversion_exclusions) == 1
    if mode.startswith("provider_failure"):
        assert all(item.status == RunStatus.FAILED and item.error for item in records)
        assert all(
            item.trace is not None and item.trace.model_requests for item in records
        )
    if not saved_count:
        assert not (artifact / "training.jsonl").read_bytes()
    before = len(activity.requests), len(activity.queries)
    assert generate.generate_dataset(config, selections, output) == artifact
    assert before == (len(activity.requests), len(activity.queries))


def test_preparation_failure_preserves_paid_work_and_input_changes_are_rejected(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    original = dataset.load_student

    def fail(config: SftExperimentConfig) -> dataset.StudentRenderer:
        raise RuntimeError("student unavailable")

    monkeypatch.setattr(dataset, "load_student", fail)
    with pytest.raises(RuntimeError, match="student unavailable"):
        dataset.prepare_dataset(config, selections, tmp_path / "dataset")
    before = len(activity.requests), len(activity.queries)
    assert len(attempts(tmp_path / "generation")) == 2
    monkeypatch.setattr(dataset, "load_student", original)
    dataset.prepare_dataset(config, selections, tmp_path / "dataset", resume=True)
    assert before == (len(activity.requests), len(activity.queries))
    assert config.data.generation is not None
    changed = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": config.data.generation.model_copy(
                        update={"plan_only": True}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="generation inputs changed"):
        generate.generate_dataset(changed, selections, tmp_path / "generation")


def test_feedback_semantic_repair_and_earlier_selection_use_recorded_requests(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    calls = [
        (
            "evaluate_candidate",
            {"action": {"version": 1, "settings": {"seq_page_cost": 2.0}}},
        ),
        # A schema-valid but impossible join subtree: retain recovery supervision.
        (
            "evaluate_candidate",
            {
                "action": {
                    "version": 1,
                    "joins": [{"relations": ["ci", "n"], "force": "nestloop"}],
                }
            },
        ),
        (
            "evaluate_candidate",
            {"action": {"version": 1, "settings": {"seq_page_cost": 3.0}}},
        ),
        ("finish", {"selected_candidate_id": "candidate-01"}),
    ]

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        assert body is not None
        if path == "responses/input_tokens":
            return {"input_tokens": 100}
        history = body["input"]
        assert isinstance(history, list)
        index = sum(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in history
        )
        name, arguments = calls[index]
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call-{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }

    monkeypatch.setattr(HttpTransport, "request", request)
    report = dataset.prepare_dataset(config, selections, tmp_path / "dataset")
    assert report.training.accepted_requests == report.validation.accepted_requests == 4
    for attempt in attempts(tmp_path / "generation"):
        assert attempt.trace is not None and attempt.rollout is not None
        assert attempt.trace.selection.selected_candidate_id == "candidate-01"
        assert len(attempt.rollout.candidates) == 3
        assert not attempt.rollout.candidates[1].constraints_satisfied
        proposal = attempt.trace.model_requests[1]
        definition = next(
            tool
            for tool in proposal.tools
            if tool.function.name == "evaluate_candidate"
        )
        validate(calls[1][1], definition.function.parameters)
        assert attempt.rollout.candidates[0].execution_feedback is not None
        for index, actual in enumerate(attempt.trace.model_requests):
            assert actual.messages == attempt.trace.transcript[: 2 + index * 2]
            assert "paired_measurements" not in actual.model_dump_json()


def test_generation_uses_existing_transport_retries_without_replacement(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    activity.mode = "keep"
    keys: dict[str, int] = {}
    bodies: dict[str, JsonObject | None] = {}

    def exchange(
        transport: HttpTransport,
        path: str,
        body: JsonObject | None,
        idempotency_key: str,
    ) -> JsonObject:
        keys[idempotency_key] = keys.get(idempotency_key, 0) + 1
        if idempotency_key not in bodies:
            bodies[idempotency_key] = body
            raise TransientModelError("temporary transport failure")
        assert bodies[idempotency_key] == body
        return activity.request(transport, path, body)

    monkeypatch.setattr(HttpTransport, "request", HTTP_REQUEST)
    monkeypatch.setattr(HttpTransport, "_request", exchange)

    def no_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr("qorl.model.client.time.sleep", no_wait)
    generate.generate_dataset(config, selections, tmp_path / "generation")
    assert len(keys) == 4  # one counting request and one completion per conversation
    assert set(keys.values()) == {2}
    assert all(
        item.trace is not None and len(item.trace.model_responses) == 1
        for item in attempts(tmp_path / "generation")
    )


def test_no_usable_split_retains_generation_and_rendering_reports(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    tmp_path: Path,
    activity: EvaluationActivity,
) -> None:
    config, selections = configured
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"context_length": 256})}
    )
    with pytest.raises(ValueError, match="empty training or validation split"):
        dataset.prepare_dataset(config, selections, tmp_path / "dataset")
    assert len(attempts(tmp_path / "generation")) == 2
    assert (tmp_path / "generation/report.json").exists()
    rendered = RenderedConversation.model_validate_json(
        (tmp_path / "dataset/training/rendered/000000.json").read_bytes()
    )
    assert rendered.rejection is not None
    assert not (tmp_path / "dataset/training/packed.jsonl").read_bytes()
    assert (tmp_path / "dataset/report.json").exists()


def test_pool_failure_records_every_requested_attempt(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(generate, "start_pool", fail)
    with pytest.raises(RuntimeError, match="pool unavailable"):
        generate.generate_dataset(config, selections, tmp_path / "generation")
    saved = attempts(tmp_path / "generation")
    assert len(saved) == 2
    assert all(
        item.status == RunStatus.FAILED
        and item.trace is None
        and item.error == "pool unavailable"
        for item in saved
    )
    report = generate.GenerationReport.model_validate_json(
        (tmp_path / "generation/report.json").read_bytes()
    )
    assert report.training.failed_attempts == report.validation.failed_attempts == 1
    assert (
        report.training.saved_conversations
        == report.validation.saved_conversations
        == 0
    )


def test_active_conversations_are_bounded_by_teacher_limit(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    assert config.data.generation is not None
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": config.data.generation.model_copy(
                        update={"generations_per_task": 2}
                    )
                }
            )
        }
    )
    activity.mode = "keep"
    activity.barrier = Barrier(2)
    observed: list[int] = []

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        observed.append(len(activity.claimed_workers))
        assert observed[-1] <= 2
        return activity.request(transport, path, body)

    monkeypatch.setattr(HttpTransport, "request", request)
    generate.generate_dataset(config, selections, tmp_path / "generation")
    assert max(observed) == 2
    assert len(attempts(tmp_path / "generation")) == 4
    assert len(activity.closed) == 1 and not activity.claimed_workers


def test_interruption_saves_attempts_and_releases_workers(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
) -> None:
    config, selections = configured
    activity.mode = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        generate.generate_dataset(config, selections, tmp_path / "generation")
    saved = attempts(tmp_path / "generation")
    assert len(saved) == 2 and all(item.status == RunStatus.FAILED for item in saved)
    assert len(activity.closed) == 1 and not activity.claimed_workers
    assert (tmp_path / "generation/report.json").exists()
    identity = (tmp_path / "generation/identity.json").read_bytes()
    before = len(activity.requests), len(activity.queries)
    generate.generate_dataset(config, selections, tmp_path / "generation")
    assert (tmp_path / "generation/identity.json").read_bytes() == identity
    assert attempts(tmp_path / "generation") == saved
    assert (len(activity.requests), len(activity.queries)) == before


def test_independent_generations_combine_without_rewriting_ids(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
) -> None:
    config, selections = configured
    activity.mode = "keep"
    first = generate.generate_dataset(config, selections, tmp_path / "first")
    second = generate.generate_dataset(config, selections, tmp_path / "second")
    first_attempts = attempts(first.parent)
    second_attempts = attempts(second.parent)
    assert {item.attempt_id for item in first_attempts}.isdisjoint(
        item.attempt_id for item in second_attempts
    )
    for left in first_attempts:
        # Random run IDs must not enter either seed derivation.
        matching = next(
            item for item in second_attempts if item.task_id == left.task_id
        )
        assert (left.model_seed, left.measurement_seed) == (
            matching.model_seed,
            matching.measurement_seed,
        )
    identities = [
        generate.GenerationIdentity.model_validate_json(
            (source.parent / "identity.json").read_bytes()
        )
        for source in (first, second)
    ]
    assert identities[0].run_id != identities[1].run_id
    assert (
        identities[0].model_copy(update={"run_id": identities[1].run_id})
        == identities[1]
    )
    before = len(activity.requests), len(activity.queries)
    generate.generate_dataset(config, selections, first.parent)
    assert attempts(first.parent) == first_attempts
    combined = tmp_path / "combined"
    combined.mkdir()
    (combined / "manifest.json").write_bytes((first / "manifest.json").read_bytes())
    for name in ("training", "validation"):
        (combined / f"{name}.jsonl").write_bytes(
            (first / f"{name}.jsonl").read_bytes()
            + (second / f"{name}.jsonl").read_bytes()
        )
    reuse = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": None,
                    "dataset_from": combined,
                    "imported_generation_seeds": ImportedGenerationSeeds(
                        training=config.experiment.seed,
                        validation=config.experiment.seed,
                    ),
                }
            )
        }
    )
    source = dataset.load_source(reuse, selections)
    assert all(len(records) == 2 for records in source.conversations.values())
    report = dataset.prepare_dataset(reuse, selections, tmp_path / "prepared")
    assert (
        report.training.accepted_conversations
        == report.validation.accepted_conversations
        == 2
    )
    assert report.training.accepted_requests == report.validation.accepted_requests == 2
    for name in ("training", "validation"):
        assert (tmp_path / f"prepared/{name}.jsonl").read_bytes() == (
            combined / f"{name}.jsonl"
        ).read_bytes()
    assert (len(activity.requests), len(activity.queries)) == before


def test_reopening_before_artifact_publication_reuses_allocated_run_id(
    configured: tuple[SftExperimentConfig, dict[TaskRole, TaskSelection]],
    activity: EvaluationActivity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selections = configured
    activity.mode = "keep"
    output = tmp_path / "generation"
    publish = generate.save_artifact

    def interrupt(*args: object, **kwargs: object) -> Path:
        raise KeyboardInterrupt()

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        identity = generate.GenerationIdentity.model_validate_json(
            (output / "identity.json").read_bytes()
        )
        assert (
            identity.run_id.version == 4
        )  # Allocated and saved before any provider work.
        return activity.request(transport, path, body)

    monkeypatch.setattr(HttpTransport, "request", request)
    monkeypatch.setattr(generate, "save_artifact", interrupt)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_dataset(config, selections, output)
    identity_bytes = (output / "identity.json").read_bytes()
    saved = attempts(output)
    assert not (output / "report.json").exists()
    before = len(activity.requests), len(activity.queries)
    monkeypatch.setattr(generate, "save_artifact", publish)
    generate.generate_dataset(config, selections, output)
    assert (output / "identity.json").read_bytes() == identity_bytes
    assert attempts(output) == saved
    assert (len(activity.requests), len(activity.queries)) == before
