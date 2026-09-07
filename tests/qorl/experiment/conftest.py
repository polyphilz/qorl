"""Small local artifacts for files-only experiment creation."""

from pathlib import Path

import pytest

from qorl.experiment import create
from qorl.experiment.schemas import CreateRequest, ExperimentMethod
from qorl.sft.schemas import PreparedDatasetManifest, PreparedDatasetSplit
from qorl.taskset.schemas import TaskRole
from qorl.taskset.selection import select_tasks
from qorl.taskset.taskset import TaskSet

TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"
SOURCE_SELECTION_SEED = 17
SOURCE_GENERATION_SEED = 23


@pytest.fixture
def experiment_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "experiments"
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", directory)
    return directory


@pytest.fixture
def creation_request(experiment_directory: Path) -> CreateRequest:
    return CreateRequest(
        name="example",
        method=ExperimentMethod.RL,
        base_model_name_or_path="example/model",
        base_model_revision=TEST_REVISION,
        tasksets=("train=ceb[2a:2]", "validation=ceb[4a:1]"),
        postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
        pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
    )


@pytest.fixture
def dataset_artifact(tmp_path: Path, benchmark_task_sets: dict[str, TaskSet]) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    selected = select_tasks(
        ["train=ceb[2a:2]", "validation=ceb[4a:1]"],
        benchmark_task_sets,
        SOURCE_SELECTION_SEED,
    )
    # Creation checks metadata and file presence; conversation validation is slice 8.
    for filename in ("training.jsonl", "validation.jsonl"):
        (directory / filename).write_text('{"messages": []}\n')
    manifest = PreparedDatasetManifest(
        schema_version=2,
        format="qorl-conversations",
        training=PreparedDatasetSplit(
            selection=selected[TaskRole.TRAIN],
            selection_seed=SOURCE_SELECTION_SEED,
            generation_seed=SOURCE_GENERATION_SEED,
            selection_expression="train=ceb[2a:2]",
            conversations=Path("training.jsonl"),
        ),
        validation=PreparedDatasetSplit(
            selection=selected[TaskRole.VALIDATION],
            selection_seed=SOURCE_SELECTION_SEED,
            generation_seed=SOURCE_GENERATION_SEED,
            selection_expression="validation=ceb[4a:1]",
            conversations=Path("validation.jsonl"),
        ),
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return directory
