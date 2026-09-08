"""Opt-in real training and adapter-serving check; no quality claims."""

import os
import re
from pathlib import Path

import pytest
import tomli_w
from pydantic import TypeAdapter
from renderers.configs import Qwen35RendererConfig

from qorl.evaluation.schemas import EvaluationReport, EvaluationSettings
from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CreateRequest,
    ExperimentMethod,
    RunRequest,
    RunStage,
    SftExperimentConfig,
    load_config,
)
from qorl.model.schemas import LocalInferenceSettings, Message, MessageRole
from qorl.sft import dataset
from qorl.sft.schemas import (
    Conversation,
    ConversationRequest,
    DatasetPreparationReport,
    PreparedDatasetManifest,
    PreparedDatasetSplit,
    SftTrainingReport,
)
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.training.schemas import CheckpointSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_TRAINING_GPU") != "1",
    reason="set QORL_TEST_TRAINING_GPU=1 on the benchmark host",
)
CONTEXT_LENGTH = 256
TRAINING_ROWS = 10
VALIDATION_ROWS = 3
FILLER_REPETITIONS = 160
BATCH_SIZE = 8
EPOCHS = 2
FINAL_UPDATE = 4
ROW_IDS = TypeAdapter(list[int])


def consumed_batches(training: Path) -> list[list[int]]:
    """Read the row IDs recorded by the actual trainer updates."""
    return [
        ROW_IDS.validate_json(rows)
        for rows in re.findall(
            r"Prepared rows \| Step \d+ \| Epoch \d+ \| Rows (\[[^\n]*\])",
            (training / "logs/latest/trainer.log").read_text(),
        )
    ]


def smoke_conversations(task_id: str, count: int) -> list[Conversation]:
    """Author plain-text mechanical examples, not optimization demonstrations."""
    return [
        Conversation(
            conversation_id=f"{task_id}-{index}",
            task_id=task_id,
            metadata={
                "purpose": "mechanical training check; not optimization training data"
            },
            requests=[ConversationRequest(assistant_message_index=2, tools=[])],
            messages=[
                Message(role=MessageRole.SYSTEM, content="Repeat the requested text."),
                Message(
                    role=MessageRole.USER, content=f"Example {index}: repeat token."
                ),
                Message(
                    role=MessageRole.ASSISTANT, content="token " * FILLER_REPETITIONS
                ),
            ],
        )
        for index in range(count)
    ]


def test_training_and_saved_adapter_evaluation(repository_root: Path) -> None:
    """Train 10 rows in 8+2 updates, then serve the saved adapter."""
    base = Path(os.environ["QORL_TEST_BASE_MODEL"])
    source = run.numbered_output(repository_root / "outputs/slice9-sources")
    dataset.write_records(
        source / "training.jsonl", smoke_conversations("job-01a", TRAINING_ROWS)
    )
    dataset.write_records(
        source / "validation.jsonl", smoke_conversations("job-02a", VALIDATION_ROWS)
    )
    manifest = PreparedDatasetManifest(
        schema_version=2,
        format="qorl-conversations",
        training=PreparedDatasetSplit(
            selection=TaskSelection(benchmark_id=BenchmarkId.JOB, task_ids=["job-01a"]),
            selection_seed=0,
            generation_seed=0,
            conversations=Path("training.jsonl"),
        ),
        validation=PreparedDatasetSplit(
            selection=TaskSelection(benchmark_id=BenchmarkId.JOB, task_ids=["job-02a"]),
            selection_seed=0,
            generation_seed=0,
            conversations=Path("validation.jsonl"),
        ),
    )
    (source / "manifest.json").write_text(manifest.model_dump_json())
    repository_root.joinpath("experiments").mkdir(exist_ok=True)
    directory = create.create_experiment(
        CreateRequest(
            name="slice9-smoke",
            method=ExperimentMethod.SFT,
            base_model_name_or_path=str(base),
            dataset_from=source.resolve(),
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, SftExperimentConfig) and isinstance(
        config.inference, LocalInferenceSettings
    )
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"context_length": CONTEXT_LENGTH}),
            "inference": config.inference.model_copy(
                update={"max_tokens": CONTEXT_LENGTH, "thinking": False}
            ),
            "training": config.training.model_copy(
                update={
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "renderer": Qwen35RendererConfig(),
                    "checkpoints": CheckpointSettings(type="interval", interval=1),
                }
            ),
        }
    )
    (directory / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    assert run.launch_experiment(directory, RunRequest(stage=RunStage.PREPARE)) == 0
    output = (run.OUTPUTS_DIRECTORY / directory.name / "000").resolve()
    prepared = DatasetPreparationReport.model_validate_json(
        (output / "dataset/report.json").read_bytes()
    )
    assert prepared.training.accepted_conversations == TRAINING_ROWS
    assert prepared.training.accepted_requests == TRAINING_ROWS
    assert prepared.training.packed_rows == TRAINING_ROWS
    assert prepared.validation.packed_rows == VALIDATION_ROWS
    assert (
        run.launch_experiment(directory, RunRequest(stage=RunStage.TRAIN, number=0))
        == 0
    )
    training = SftTrainingReport.model_validate_json(
        (output / "training/report.json").read_bytes()
    )
    assert training.optimizer_updates == FINAL_UPDATE
    assert [loss.step for loss in training.validation_losses] == [0, 2, 4]
    batches = consumed_batches(output / "training")
    assert [len(rows) for rows in batches] == [8, 2, 8, 2]
    assert sorted(batches[0] + batches[1]) == list(range(TRAINING_ROWS))
    assert sorted(batches[2] + batches[3]) == list(range(TRAINING_ROWS))

    # The mechanical training uses short sequences; a separate evaluation uses
    # the normal agent context length and these explicitly selected saved weights.
    evaluation = create.create_experiment(
        CreateRequest(
            name="slice9-adapter-eval",
            method=ExperimentMethod.EVAL,
            base_model_name_or_path=str(base),
            adapter_path=training.final_adapter,
            tasksets=("test=job[01:1]",),
            postgres_config=config.postgres.path,
            pool_config=config.pool.path,
        )
    )
    evaluation_config = load_config(evaluation / "config.toml")
    evaluation_config = evaluation_config.model_copy(
        update={"evaluation": EvaluationSettings(rollouts_per_task=1)}
    )
    (evaluation / "config.toml").write_text(
        tomli_w.dumps(evaluation_config.model_dump(mode="json", exclude_none=True))
    )
    assert (
        run.launch_experiment(
            evaluation, RunRequest(stage=RunStage.EVALUATE, split=TaskRole.TEST)
        )
        == 0
    )
    result_path = (
        run.OUTPUTS_DIRECTORY
        / evaluation.name
        / "000/evaluation/test/000/evaluation.json"
    )
    result = EvaluationReport.model_validate_json(result_path.read_bytes())
    assert result.model.adapter_path == training.final_adapter
    assert result.summary.recorded_rollout_count == 1
    assert result.summary.performance.failure_count == 0
