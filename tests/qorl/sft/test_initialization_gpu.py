"""Opt-in real SFT initialization on new rows; source adapter remains untouched."""

import os
from pathlib import Path

import pytest
import torch
from renderers.configs import Qwen35RendererConfig
from safetensors.torch import load as load_tensors
from tests.qorl.sft.test_training_gpu import consumed_batches, smoke_conversations

from qorl.experiment.schemas import SftExperimentConfig, load_config
from qorl.sft import dataset, train
from qorl.sft.schemas import (
    ImportedGenerationSeeds,
    PreparedDatasetManifest,
    PreparedDatasetSplit,
    TrainingIdentity,
)
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_INITIALIZATION_GPU") != "1",
    reason="set QORL_TEST_INITIALIZATION_GPU=1 in an isolated GPU checkout",
)


def test_initial_adapter_on_new_dataset(repository_root: Path) -> None:
    evidence = Path(os.environ["QORL_INITIALIZATION_EVIDENCE"])
    evidence.mkdir(parents=True, exist_ok=False)
    config = load_config(repository_root / "configs/defaults/000-sft.toml")
    assert isinstance(config, SftExperimentConfig)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "name_or_path": os.environ["QORL_TEST_BASE_MODEL"],
                    "revision": None,
                    "context_length": 256,
                }
            ),
            "inference": config.inference.model_copy(
                update={"max_tokens": 256, "thinking": False}
            ),
            "training": config.training.model_copy(
                update={
                    "epochs": 1,
                    "batch_size": 8,
                    "renderer": Qwen35RendererConfig(),
                }
            ),
        }
    )
    selections = {
        TaskRole.TRAIN: TaskSelection(
            benchmark_id=BenchmarkId.JOB, task_ids=["job-01a"]
        ),
        TaskRole.VALIDATION: TaskSelection(
            benchmark_id=BenchmarkId.JOB, task_ids=["job-02a"]
        ),
    }

    def prepare(name: str, count: int) -> tuple[SftExperimentConfig, Path]:
        source = evidence / f"{name}-conversations"
        for role, split, size in (
            (TaskRole.TRAIN, "training", count),
            (TaskRole.VALIDATION, "validation", 3),
        ):
            dataset.write_records(
                source / f"{split}.jsonl",
                smoke_conversations(selections[role].task_ids[0], size),
            )
        write_json(
            source / "manifest.json",
            PreparedDatasetManifest(
                schema_version=2,
                format="qorl-conversations",
                training=PreparedDatasetSplit(
                    selection=selections[TaskRole.TRAIN],
                    selection_seed=42,
                    selection_expression=config.data.training.expression,
                    generation_seed=42,
                    conversations=Path("training.jsonl"),
                ),
                validation=PreparedDatasetSplit(
                    selection=selections[TaskRole.VALIDATION],
                    selection_seed=42,
                    selection_expression=config.data.validation.expression,
                    generation_seed=42,
                    conversations=Path("validation.jsonl"),
                ),
            ).model_dump(mode="json"),
        )
        selected = config.model_copy(
            update={
                "data": config.data.model_copy(
                    update={
                        "dataset_from": source,
                        "generation": None,
                        "imported_generation_seeds": ImportedGenerationSeeds(
                            training=42, validation=42
                        ),
                    }
                )
            }
        )
        output = evidence / name
        output.mkdir()
        report = dataset.prepare_dataset(selected, selections, output / "dataset")
        assert report.training.packed_rows == count
        return selected, output

    source_config, source_run = prepare("source", 10)
    first = train.train(source_config, source_run)
    original = {
        path: sha256_file(path) for path in source_run.rglob("*") if path.is_file()
    }
    destination_config, destination_run = prepare("destination", 13)
    second = train.train(
        destination_config, destination_run, init_adapter=first.final_adapter
    )
    assert second.starting_step == 0 and second.optimizer_updates == 2
    assert second.validation_losses[0].step == 0
    assert second.validation_losses[0].loss == pytest.approx(
        first.validation_losses[-1].loss, rel=1e-5
    )
    assert [len(rows) for rows in consumed_batches(destination_run / "training")] == [
        8,
        5,
    ]
    assert original == {path: sha256_file(path) for path in original}
    before = load_tensors(
        (first.final_adapter / "adapter_model.safetensors").read_bytes()
    )
    after = load_tensors(
        (second.final_adapter / "adapter_model.safetensors").read_bytes()
    )
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    identity = TrainingIdentity.model_validate_json(
        (destination_run / "training" / train.TRAINING_IDENTITY).read_bytes()
    )
    assert identity.initialization is not None and identity.continuation is None
    assert identity.initialization.adapter_sha256 == sha256_file(
        first.final_adapter / "adapter_model.safetensors"
    )
    write_json(
        evidence / "verification.json",
        {
            "source": first.model_dump(mode="json"),
            "destination": second.model_dump(mode="json"),
            "source_unchanged": True,
            "adapter_updated": True,
        },
    )
