"""Opt-in native completed-epoch continuation; no teacher, SQL or quality claims."""

import json
import os
from pathlib import Path

import pytest
import tomli_w
import torch
from safetensors.torch import load as load_tensors
from tests.qorl.sft.test_training_gpu import consumed_batches, smoke_conversations

from qorl.experiment import create
from qorl.experiment.schemas import (
    CreateRequest,
    ExperimentMethod,
    SftExperimentConfig,
    load_config,
)
from qorl.sft import dataset, train
from qorl.sft.schemas import PreparedDatasetManifest, PreparedDatasetSplit
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.training.schemas import CheckpointSettings
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_CONTINUATION_GPU") != "1",
    reason="set QORL_TEST_CONTINUATION_GPU=1 in an isolated benchmark-host checkout",
)


def test_completed_epoch_continuation() -> None:
    evidence = Path(os.environ["QORL_CONTINUATION_EVIDENCE"])
    evidence.mkdir(parents=True, exist_ok=False)
    source = evidence / "conversations"
    selections = {
        TaskRole.TRAIN: TaskSelection(
            benchmark_id=BenchmarkId.JOB, task_ids=["job-01a"]
        ),
        TaskRole.VALIDATION: TaskSelection(
            benchmark_id=BenchmarkId.JOB, task_ids=["job-02a"]
        ),
    }
    for role, name, count in (
        (TaskRole.TRAIN, "training", 10),
        (TaskRole.VALIDATION, "validation", 3),
    ):
        dataset.write_records(
            source / f"{name}.jsonl",
            smoke_conversations(selections[role].task_ids[0], count),
        )
    write_json(
        source / "manifest.json",
        PreparedDatasetManifest(
            schema_version=2,
            format="qorl-conversations",
            training=PreparedDatasetSplit(
                selection=selections[TaskRole.TRAIN],
                selection_seed=0,
                generation_seed=0,
                conversations=Path("training.jsonl"),
            ),
            validation=PreparedDatasetSplit(
                selection=selections[TaskRole.VALIDATION],
                selection_seed=0,
                generation_seed=0,
                conversations=Path("validation.jsonl"),
            ),
        ).model_dump(mode="json"),
    )
    directory = create.create_experiment(
        CreateRequest(
            name="continuation-smoke",
            method=ExperimentMethod.SFT,
            base_model_name_or_path=os.environ["QORL_TEST_BASE_MODEL"],
            dataset_from=source,
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
        )
    )
    config = load_config(directory / "config.toml")
    assert isinstance(config, SftExperimentConfig)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"context_length": 256}),
            "inference": config.inference.model_copy(
                update={"max_tokens": 256, "thinking": False}
            ),
            "training": config.training.model_copy(
                update={
                    "epochs": 2,
                    "batch_size": 8,
                    "checkpoints": CheckpointSettings(type="interval", interval=2),
                }
            ),
        }
    )

    def prepare(name: str, epochs: int) -> tuple[SftExperimentConfig, Path]:
        selected = config.model_copy(
            update={"training": config.training.model_copy(update={"epochs": epochs})}
        )
        output = evidence / name
        output.mkdir()
        (output / "config.toml").write_text(
            tomli_w.dumps(selected.model_dump(mode="json", exclude_none=True))
        )
        report = dataset.prepare_dataset(selected, selections, output / "dataset")
        assert report.training.packed_rows == 10
        return selected, output

    complete_config, complete_run = prepare("uninterrupted", 2)
    complete = train.train(complete_config, complete_run)
    first_config, first_run = prepare("source", 1)
    first = train.train(first_config, first_run)
    source_hashes = {
        str(path.relative_to(first_run)): sha256_file(path)
        for path in first_run.rglob("*")
        if path.is_file()
    }
    write_json(evidence / "source-before.json", source_hashes)
    resumed_config, resumed_run = prepare("destination", 2)
    resumed = train.train(
        resumed_config, resumed_run, resume_from=first.checkpoints[-1].parent
    )
    assert source_hashes == {
        str(path.relative_to(first_run)): sha256_file(path)
        for path in first_run.rglob("*")
        if path.is_file()
    }
    assert complete.optimizer_updates == resumed.optimizer_updates == 4
    assert resumed.starting_step == 2
    assert [loss.step for loss in resumed.validation_losses] == [2, 4]
    assert resumed.validation_losses[0].loss == pytest.approx(
        first.validation_losses[-1].loss, rel=1e-5
    )
    complete_rows = consumed_batches(complete_run / "training")
    assert (
        consumed_batches(first_run / "training")
        + consumed_batches(resumed_run / "training")
        == complete_rows
    )
    assert [len(rows) for rows in complete_rows] == [8, 2, 8, 2]
    expected = load_tensors(
        (complete.final_adapter / "adapter_model.safetensors").read_bytes()
    )
    actual = load_tensors(
        (resumed.final_adapter / "adapter_model.safetensors").read_bytes()
    )
    assert actual.keys() == expected.keys()
    prefix_model = train.checkpoint_model(
        complete_config.model,
        complete_run / "training",
        complete_run / "training/checkpoints/step_2",
    )
    assert prefix_model.adapter_path is not None
    prefix_expected = load_tensors(
        (prefix_model.adapter_path / "adapter_model.safetensors").read_bytes()
    )
    prefix_actual = load_tensors(
        (first.final_adapter / "adapter_model.safetensors").read_bytes()
    )
    prefix_differences: dict[str, float] = {}
    relative_differences: dict[str, float] = {}
    for name in actual:
        # Separate BF16 runs differ even before loading; strict FP32 state/update
        # equivalence is covered by the native checkpoint test.
        relative_differences[name] = float(
            (actual[name].float() - expected[name].float()).square().sum().sqrt()
            / expected[name].float().square().sum().sqrt()
        )
        prefix_differences[name] = float(
            (prefix_actual[name].float() - prefix_expected[name].float())
            .square()
            .sum()
            .sqrt()
            / prefix_expected[name].float().square().sum().sqrt()
        )
        # Empirical smoke allowance, not bitwise replay: twice the measured
        # independent-run discrepancy plus two BF16 rounding units.
        assert (
            relative_differences[name]
            <= 2 * prefix_differences[name] + 2 * torch.finfo(actual[name].dtype).eps
        )
    assert resumed.validation_losses[-1].loss == pytest.approx(
        complete.validation_losses[-1].loss, rel=0.01
    )
    resolved = train.checkpoint_model(
        resumed_config.model, resumed_run / "training", resumed.checkpoints[-1]
    )
    assert resolved.adapter_path == resumed.final_adapter
    write_json(evidence / "source-files-unchanged.json", source_hashes)
    write_json(evidence / "adapter-relative-differences.json", relative_differences)
    write_json(evidence / "prefix-relative-differences.json", prefix_differences)
    (evidence / "row-order.json").write_text(json.dumps(complete_rows))
    write_json(evidence / "verification.json", resumed.model_dump(mode="json"))
    write_json(
        evidence / "adapter-max-difference.json",
        max(
            float((actual[name].float() - expected[name].float()).abs().max())
            for name in actual
        ),
    )
