"""Opt-in two-conversation Astra generation gate; no training or quality claim."""

import os
import tomllib
from pathlib import Path

import pytest
import tomli_w
from renderers.configs import Qwen35RendererConfig

from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CreateRequest,
    ExperimentMethod,
    RunRequest,
    RunStage,
    SftExperimentConfig,
    load_config,
)
from qorl.model.schemas import (
    AstraInferenceSettings,
    ModelPreset,
    ReasoningEffort,
    ToolDefinition,
)
from qorl.sft import dataset
from qorl.sft.generate import GenerationAttempt, GenerationReport
from qorl.sft.schemas import (
    Conversation,
    DatasetPreparationReport,
    GenerationSettings,
    RenderedConversation,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_GENERATION_LIVE") != "1",
    reason="set QORL_TEST_GENERATION_LIVE=1 for two paid Astra conversations on the benchmark host",
)


def verify_saved_requests(output: Path) -> int:
    """Inspect a retained live run without provider calls or SQL."""
    verified = 0
    for name in ("training", "validation"):
        conversation = Conversation.model_validate_json(
            (output / f"dataset/{name}.jsonl").read_bytes()
        )
        attempt = GenerationAttempt.model_validate_json(
            (
                output / "generation/attempts" / f"{conversation.conversation_id}.json"
            ).read_bytes()
        )
        assert attempt.trace is not None
        rendered = RenderedConversation.model_validate_json(
            (output / f"dataset/{name}/rendered/000000.json").read_bytes()
        )
        assert rendered.rejection is None
        assert conversation.messages == attempt.trace.transcript
        dataset.validate_conversation(conversation)
        for request, original, sample in zip(
            conversation.requests,
            attempt.trace.model_requests,
            rendered.requests,
            strict=True,
        ):
            assert (
                conversation.messages[: request.assistant_message_index]
                == original.messages
            )
            assert request.tools == original.tools
            assert sample.assistant_message_index == request.assistant_message_index
            definitions = sample.rendered_text.split("<tools>\n")[1].split(
                "\n</tools>"
            )[0]
            assert [
                ToolDefinition.model_validate_json(line)
                for line in definitions.splitlines()
            ] == original.tools
            assert any(sample.sample.loss_mask)
            assert all(
                index == request.assistant_message_index
                for index, mask in zip(
                    sample.target_message_indices, sample.sample.loss_mask, strict=True
                )
                if mask
            )
            verified += 1
    return verified


def test_saved_tasks_to_astra_to_student_tokens(
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = Path(os.environ["QORL_TEST_GENERATION_OUTPUT"])
    evidence.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", evidence / "experiments")
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", evidence / "runs")
    directory = create.create_experiment(
        CreateRequest(
            name="astra-generation",
            method=ExperimentMethod.SFT,
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
            tasksets=("train=job[01:1]", "validation=job[02:1]"),
            base_model_name_or_path=os.environ["QORL_TEST_BASE_MODEL"],
            seed=42,
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, SftExperimentConfig)
    teacher = ModelPreset.model_validate(
        tomllib.loads(
            (
                repository_root / "configs/defaults/models/000-gpt-6-astra.toml"
            ).read_text()
        )
    )
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "generation": GenerationSettings(
                        model=teacher.model.model_copy(
                            update={"max_concurrent_requests": 2}
                        ),
                        inference=AstraInferenceSettings(
                            max_tokens=4096, reasoning_effort=ReasoningEffort.MEDIUM
                        ),
                        generations_per_task=1,
                        plan_only=False,
                    )
                }
            ),
            "agent": config.agent.model_copy(
                update={"candidate_attempts": 2, "maximum_model_turns": 6}
            ),
            "training": config.training.model_copy(
                update={"renderer": Qwen35RendererConfig()}
            ),
        }
    )
    (directory / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    output = run.run_experiment(directory, RunRequest(stage=RunStage.PREPARE))
    generated = GenerationReport.model_validate_json(
        (output / "generation/report.json").read_bytes()
    )
    prepared = DatasetPreparationReport.model_validate_json(
        (output / "dataset/report.json").read_bytes()
    )
    for generation, preparation in (
        (generated.training, prepared.training),
        (generated.validation, prepared.validation),
    ):
        assert (
            generation.requested_attempts
            == generation.completed_attempts + generation.failed_attempts
            == 1
        )
        assert (
            generation.saved_conversations + len(generation.conversion_exclusions) == 1
        )
        assert preparation.source_conversations == generation.saved_conversations
        assert preparation.accepted_requests > 0 and preparation.packed_rows > 0
    for path in (output / "generation/attempts").glob("*.json"):
        attempt = GenerationAttempt.model_validate_json(path.read_bytes())
        assert attempt.trace is not None and attempt.rollout is not None
        assert attempt.trace.model_requests and attempt.trace.model_responses
    assert (
        verify_saved_requests(output)
        == prepared.training.accepted_requests + prepared.validation.accepted_requests
    )
    # Reuse runs through ordinary creation, preserving IDs, requests, provenance and seeds.
    reused = create.create_experiment(
        CreateRequest(
            name="astra-reused",
            method=ExperimentMethod.SFT,
            postgres_config=config.postgres.path,
            pool_config=config.pool.path,
            base_model_name_or_path=config.model.name_or_path,
            dataset_from=output / "dataset",
            seed=43,
        )
    )
    reused_config = load_config(reused / "config.toml")
    assert isinstance(reused_config, SftExperimentConfig)
    reused_config = reused_config.model_copy(update={"training": config.training})
    (reused / "config.toml").write_text(
        tomli_w.dumps(reused_config.model_dump(mode="json", exclude_none=True))
    )
    from qorl.sft import generate

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset-from must not generate or use PostgreSQL")

    monkeypatch.setattr(generate, "generate_dataset", forbidden)
    reused_output = run.run_experiment(reused, RunRequest(stage=RunStage.PREPARE))
    for name in ("training", "validation"):
        assert (output / f"dataset/{name}.jsonl").read_bytes() == (
            reused_output / f"dataset/{name}.jsonl"
        ).read_bytes()
