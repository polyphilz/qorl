"""SFT configuration translation, fixed run inputs and explicit checkpoint handoff."""

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import pytest
import tomli_w
import torch
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import CosineSchedulerConfig
from pydantic import JsonValue
from safetensors.torch import save as serialize_safetensors
from torch.distributed.checkpoint import state_dict_saver
from torch.distributed.checkpoint.metadata import Metadata

from qorl.adapters.schemas import AdapterConfig
from qorl.experiment.schemas import SftExperimentConfig, load_config
from qorl.model.files import model_weights_sha256
from qorl.sft import train
from qorl.sft.schemas import (
    DatasetPreparationIdentity,
    DatasetPreparationReport,
    PreparedSplitReport,
    SftTrainingReport,
    TrainingIdentity,
    TrainingRow,
)
from qorl.training.schemas import CheckpointSettings
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json

ROW_COUNT = 10
BATCH_SIZE = 8
CONTEXT_LENGTH = 4
EPOCHS = 2
LORA_RANK = 1
LORA_ALPHA = 7.0


class CheckpointSaver(Protocol):
    def save(
        self,
        state_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
        *,
        checkpoint_id: Path,
    ) -> Metadata: ...


CHECKPOINT_SAVER: CheckpointSaver = state_dict_saver


@pytest.fixture
def config(tmp_path: Path, repository_root: Path) -> SftExperimentConfig:
    loaded = load_config(repository_root / "configs/defaults/000-sft.toml")
    assert isinstance(loaded, SftExperimentConfig)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"test base")
    return loaded.model_copy(
        update={
            "inference": loaded.inference.model_copy(
                update={"max_tokens": CONTEXT_LENGTH}
            ),
            "model": loaded.model.model_copy(
                update={
                    "name_or_path": str(base),
                    "revision": None,
                    "context_length": CONTEXT_LENGTH,
                }
            ),
            "training": loaded.training.model_copy(
                update={
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lora": loaded.training.lora.model_copy(
                        update={"rank": LORA_RANK, "alpha": LORA_ALPHA}
                    ),
                }
            ),
        }
    )


@pytest.fixture
def prepared(config: SftExperimentConfig, tmp_path: Path) -> Path:
    run = tmp_path / "run"
    dataset = run / "dataset"
    rows = [
        TrainingRow(
            input_ids=[index, 1, 2, 3],
            target_ids=[1, 2, 3, 0],
            position_ids=[0, 1, 0, 1],
            loss_mask=[False, True, False, True],
            seq_lens=[2, 2],
        )
        for index in range(ROW_COUNT)
    ]
    files: dict[str, str] = {}
    for role in ("training", "validation"):
        path = dataset / role / "packed.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
        files[path.relative_to(dataset).as_posix()] = sha256_file(path)
    split = PreparedSplitReport(
        selected_tasks=ROW_COUNT,
        source_conversations=ROW_COUNT,
        accepted_conversations=ROW_COUNT,
        accepted_requests=ROW_COUNT,
        accepted_tasks=ROW_COUNT,
        rejections={},
        packed_rows=ROW_COUNT,
        input_tokens=ROW_COUNT * CONTEXT_LENGTH,
        supervised_tokens=ROW_COUNT * 2,
        padding_tokens=0,
    )
    report = DatasetPreparationReport(
        context_length=CONTEXT_LENGTH,
        shuffle_seed=config.experiment.seed,
        training=split,
        validation=split,
        files=files,
    )
    identity = DatasetPreparationIdentity(
        config_sha256=sha256_json(config.model_dump(mode="json")),
        source=dataset,
        source_files={},
        tokenizer_sha256="test",
        renderer={},
        dependencies={},
    )
    write_json(dataset / "report.json", report.model_dump(mode="json"))
    write_json(dataset / "preparation.json", identity.model_dump(mode="json"))
    return run


@pytest.mark.parametrize("max_grad_norm", [None, 0.0, 1.0])
def test_translate_finite_schedule_and_explicit_settings(
    config: SftExperimentConfig, prepared: Path, max_grad_norm: float | None
) -> None:
    report = train.prepared_report(config, prepared / "dataset")
    changed = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"max_grad_norm": max_grad_norm}
            )
        }
    )
    native = train.trainer_config(changed, prepared, report)
    assert native.data.type == "prepared"
    assert native.data.batch_size == BATCH_SIZE
    assert native.data.epochs == EPOCHS
    assert native.max_steps == 4
    assert native.optim.max_norm == max_grad_norm
    assert (
        native.val is not None and native.val.interval == 2 and native.val.eval_on_start
    )
    assert native.model.lora is not None and native.model.lora.alpha == LORA_ALPHA
    assert native.model.optimization_dtype == config.training.runtime.optimization_dtype
    assert native.model.reduce_dtype == config.training.runtime.reduce_dtype
    assert native.model.compile is None
    assert native.run_dir == prepared / "training"
    assert SFTConfig.model_validate_json(native.model_dump_json()) == native


def test_native_scheduler_rejects_warmup_beyond_row_schedule(
    config: SftExperimentConfig, prepared: Path
) -> None:
    report = train.prepared_report(config, prepared / "dataset")
    changed = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "optimizer": config.training.optimizer.model_copy(
                        update={
                            "scheduler": CosineSchedulerConfig(
                                warmup_steps=100, min_lr=0.0
                            )
                        }
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="warmup"):
        train.trainer_config(changed, prepared, report)


@pytest.mark.parametrize(
    ("checkpoints", "expected_keep_last"),
    [
        (CheckpointSettings(type="final"), None),
        (CheckpointSettings(type="interval", interval=1), None),
        (CheckpointSettings(type="interval", interval=1, keep_interval=3), 1),
        (CheckpointSettings(type="interval", interval=1, keep_last=2), 2),
        (
            CheckpointSettings(
                type="interval", interval=1, keep_last=2, keep_interval=3
            ),
            2,
        ),
    ],
)
def test_translated_retention_keeps_latest_checkpoint(
    config: SftExperimentConfig,
    prepared: Path,
    checkpoints: CheckpointSettings,
    expected_keep_last: int | None,
) -> None:
    report = train.prepared_report(config, prepared / "dataset")
    changed = config.model_copy(
        update={
            "training": config.training.model_copy(update={"checkpoints": checkpoints})
        }
    )
    native = train.trainer_config(changed, prepared, report)
    assert native.max_steps == 4
    assert native.ckpt is not None
    assert native.ckpt.keep_last == expected_keep_last
    assert native.ckpt.keep_interval == checkpoints.keep_interval
    assert native.ckpt.interval == checkpoints.interval


def test_changed_prepared_rows_are_rejected(
    config: SftExperimentConfig, prepared: Path
) -> None:
    with (prepared / "dataset/training/packed.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="file changed"):
        train.train(config, prepared)


def test_existing_training_is_not_overwritten(
    config: SftExperimentConfig, prepared: Path
) -> None:
    (prepared / "training").mkdir()
    with pytest.raises(ValueError, match="output exists"):
        train.train(config, prepared)


@pytest.fixture
def continuation(
    config: SftExperimentConfig,
    prepared: Path,
    trained: SftTrainingReport,
    tmp_path: Path,
) -> tuple[SftExperimentConfig, Path, Path]:
    (prepared / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    destination = tmp_path / "destination"
    shutil.copytree(prepared / "dataset", destination / "dataset")
    changed = config.model_copy(
        update={"training": config.training.model_copy(update={"epochs": 3})}
    )
    identity_path = destination / "dataset/preparation.json"
    identity = DatasetPreparationIdentity.model_validate_json(
        identity_path.read_bytes()
    )
    write_json(
        identity_path,
        identity.model_copy(
            update={"config_sha256": sha256_json(changed.model_dump(mode="json"))}
        ).model_dump(mode="json"),
    )
    return changed, destination, trained.checkpoints[-1].parent


def test_continue_completed_epochs_exports_only_destination(
    continuation: tuple[SftExperimentConfig, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, destination, checkpoint = continuation
    before = {
        path: sha256_file(path)
        for path in checkpoint.parent.parent.parent.rglob("*")
        if path.is_file()
    }

    def launch(
        command: list[str], *, check: bool, env: dict[str, str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        native = SFTConfig.model_validate_json(Path(command[-1]).read_bytes())
        assert native.resume is not None and native.resume.dir is not None
        assert native.max_steps in (6, 8)
        assert native.ckpt is not None and native.ckpt.output_dir is None
        shutil.copytree(
            native.resume.dir / "trainer",
            native.run_dir / f"checkpoints/step_{native.max_steps}/trainer",
        )
        metrics = native.run_dir / "monitors/file/metrics.jsonl"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(
            "".join(
                f'{{"step":{step},"val/loss":1.0}}\n'
                for step in (native.resume.dir_step, native.max_steps)
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(train.subprocess, "run", launch)
    result = train.train(config, destination, resume_from=checkpoint)
    assert result.starting_step == 4 and result.optimizer_updates == 6
    assert [loss.step for loss in result.validation_losses] == [4, 6]
    assert result.final_adapter == destination / "training/checkpoints/step_6/adapter"
    assert before == {path: sha256_file(path) for path in before}
    assert (
        train.checkpoint_model(
            config.model, destination / "training", result.checkpoints[-1]
        ).adapter_path
        == result.final_adapter
    )
    (destination / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    next_run = destination.parent / "next-destination"
    shutil.copytree(destination / "dataset", next_run / "dataset")
    next_config = config.model_copy(
        update={"training": config.training.model_copy(update={"epochs": 4})}
    )
    identity_path = next_run / "dataset/preparation.json"
    identity = DatasetPreparationIdentity.model_validate_json(
        identity_path.read_bytes()
    )
    write_json(
        identity_path,
        identity.model_copy(
            update={"config_sha256": sha256_json(next_config.model_dump(mode="json"))}
        ).model_dump(mode="json"),
    )
    chained = train.train(
        next_config, next_run, resume_from=result.checkpoints[-1].parent
    )
    assert chained.starting_step == 6 and chained.optimizer_updates == 8
    assert [loss.step for loss in chained.validation_losses] == [6, 8]


@pytest.mark.parametrize(
    "change",
    [
        "rows",
        "optimizer",
        "model",
        "tokenizer",
        "epochs",
        "dropout",
        "seed",
        "checkpoint",
        "adapter",
        "symlink",
    ],
)
def test_continuation_rejects_incompatible_source(
    continuation: tuple[SftExperimentConfig, Path, Path], change: str
) -> None:
    config, destination, checkpoint = continuation
    report = train.prepared_report(config, destination / "dataset")
    native = train.trainer_config(config, destination, report)
    if change == "rows":
        (destination / "dataset/training/packed.jsonl").write_text("changed")
    elif change == "optimizer":
        native.optim.lr *= 2
    elif change == "model":
        native.model.seq_len += 1
    elif change == "tokenizer":
        native.tokenizer.chat_template = "changed"
    elif change == "epochs":
        config = config.model_copy(
            update={"training": config.training.model_copy(update={"epochs": 2})}
        )
    elif change == "dropout":
        config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "lora": config.training.lora.model_copy(update={"dropout": 0.1})
                    }
                )
            }
        )
    elif change == "seed":
        native.data.seed += 1
    elif change == "checkpoint":
        next((checkpoint / "trainer").glob("*.distcp")).write_bytes(b"changed")
    elif change == "adapter":
        (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"changed")
    elif change == "symlink":
        alias = destination.parent / "source-alias"
        alias.symlink_to(checkpoint.parent.parent.parent, target_is_directory=True)
        destination = (alias / "nested-destination").resolve()
    with pytest.raises(ValueError):
        train.continuation_source(config, destination, native, checkpoint)
    assert not (destination / "training").exists()


def test_relocated_continuation_and_checkpoint_keep_content_checks(
    continuation: tuple[SftExperimentConfig, Path, Path],
) -> None:
    config, destination, checkpoint = continuation
    report = train.prepared_report(config, destination / "dataset")
    source = checkpoint.parents[2]
    recorded = {
        path.relative_to(source): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    relocated = source.with_name("relocated-source")
    shutil.move(source, relocated)
    checkpoint = relocated / checkpoint.relative_to(source)
    base = Path(config.model.name_or_path)
    relocated_base = base.with_name("relocated-base")
    shutil.move(base, relocated_base)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"name_or_path": str(relocated_base)}
            )
        }
    )
    native = train.trainer_config(config, destination, report)
    result = train.continuation_source(config, destination, native, checkpoint)
    assert result.checkpoint == checkpoint
    assert result.completed_steps == 4
    model = train.checkpoint_model(config.model, relocated / "training", checkpoint)
    assert model.name_or_path == str(relocated_base)
    assert model.adapter_path == checkpoint / "adapter"
    assert recorded == {
        path.relative_to(relocated): sha256_file(path)
        for path in relocated.rglob("*")
        if path.is_file()
    }
    (relocated_base / "model.safetensors").write_bytes(b"different base")
    with pytest.raises(ValueError, match="base weights differ"):
        train.continuation_source(config, destination, native, checkpoint)
    with pytest.raises(ValueError, match="base weights changed"):
        train.checkpoint_model(config.model, relocated / "training", checkpoint)


@pytest.fixture
def trained(
    config: SftExperimentConfig,
    prepared: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SftTrainingReport:
    checkpoint = tmp_path / "native-checkpoint"
    CHECKPOINT_SAVER.save(
        {
            "app": {
                "model": {
                    "layer.q_proj.lora_A.0": torch.ones(LORA_RANK, CONTEXT_LENGTH),
                    "layer.q_proj.lora_B.0": torch.ones(CONTEXT_LENGTH, LORA_RANK),
                },
                "optimizers": {
                    "state.weight.step": torch.tensor(4),
                    "state.weight.exp_avg": torch.ones(1),
                    "state.weight.exp_avg_sq": torch.ones(1),
                },
                "scheduler": {"last_epoch": torch.tensor(4)},
                "progress": {"step": torch.tensor(4)},
            }
        },
        checkpoint_id=checkpoint,
    )

    def launch(
        command: list[str], *, check: bool, env: dict[str, str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert "prime_rl.entrypoints.sft" in command
        assert check and env["CUDA_VISIBLE_DEVICES"] == "0"
        native = SFTConfig.model_validate_json(Path(command[-1]).read_bytes())
        final = native.run_dir / "checkpoints/step_4/trainer"
        shutil.copytree(checkpoint, final)
        metrics = native.run_dir / "monitors/file/metrics.jsonl"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(
            "\n".join(f'{{"step": {step}, "val/loss": 1.0}}' for step in (0, 2, 4))
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(train.subprocess, "run", launch)
    return train.train(config, prepared)


def test_train_records_counts_and_exports_recorded_scale(
    config: SftExperimentConfig,
    prepared: Path,
    tmp_path: Path,
    trained: SftTrainingReport,
) -> None:
    result = trained
    assert result.optimizer_updates == 4 and result.steps_per_epoch == 2
    assert [loss.step for loss in result.validation_losses] == [0, 2, 4]
    assert result.training.packed_rows == ROW_COUNT
    adapter = AdapterConfig.model_validate_json(
        (result.final_adapter / "adapter_config.json").read_bytes()
    )
    assert adapter.lora_alpha == LORA_ALPHA
    assert adapter.base_model_name_or_path == config.model.name_or_path
    identity = TrainingIdentity.model_validate_json(
        (prepared / "training" / train.TRAINING_IDENTITY).read_bytes()
    )
    assert identity.base_weights_sha256 == model_weights_sha256(
        Path(config.model.name_or_path)
    )
    resolved = train.checkpoint_model(
        config.model, prepared / "training", result.checkpoints[-1].parent
    )
    assert resolved.adapter_path == result.final_adapter
    with pytest.raises(ValueError, match="already complete"):
        train.train(config, prepared)
    with pytest.raises(ValueError, match="from this run"):
        train.checkpoint_model(
            config.model, prepared / "training", tmp_path / "native-checkpoint"
        )
    shard = next(result.checkpoints[-1].glob("*.distcp"))
    with shard.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="different checkpoint weights"):
        train.checkpoint_model(
            config.model, prepared / "training", result.checkpoints[-1]
        )


def test_cached_adapter_rejects_changed_tensors(
    config: SftExperimentConfig, prepared: Path, trained: SftTrainingReport
) -> None:
    weights = trained.final_adapter / "adapter_model.safetensors"
    weights.write_bytes(
        serialize_safetensors(
            {
                "layer.q_proj.lora_A.weight": torch.full(
                    (LORA_RANK, CONTEXT_LENGTH), 2.0
                ),
                "layer.q_proj.lora_B.weight": torch.full(
                    (CONTEXT_LENGTH, LORA_RANK), 2.0
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="adapter weights changed"):
        train.checkpoint_model(
            config.model, prepared / "training", trained.checkpoints[-1]
        )


def test_initial_adapter_allows_new_data_with_fresh_progress(
    config: SftExperimentConfig,
    prepared: Path,
    trained: SftTrainingReport,
    tmp_path: Path,
) -> None:
    source = trained.final_adapter
    before = {path: sha256_file(path) for path in source.iterdir()}
    destination = tmp_path / "new-data-run"
    shutil.copytree(prepared / "dataset", destination / "dataset")
    rows = destination / "dataset/training/packed.jsonl"
    rows.write_text(rows.read_text().replace('"input_ids":[0,', '"input_ids":[99,'))
    assert sha256_file(rows) != sha256_file(prepared / "dataset/training/packed.jsonl")
    report_path = destination / "dataset/report.json"
    report = DatasetPreparationReport.model_validate_json(report_path.read_bytes())
    report.files["training/packed.jsonl"] = sha256_file(rows)
    write_json(report_path, report.model_dump(mode="json"))
    result = train.train(config, destination, init_adapter=source)
    native = SFTConfig.model_validate_json(
        (destination / "training" / train.TRAINER_CONFIG).read_bytes()
    )
    identity = TrainingIdentity.model_validate_json(
        (destination / "training" / train.TRAINING_IDENTITY).read_bytes()
    )
    assert native.resume is None
    assert native.initial_adapter == destination / "training/configs/initial_adapter"
    assert native.initial_adapter is not None
    assert identity.initialization is not None
    assert identity.initialization.adapter == source
    assert identity.continuation is None
    assert result.starting_step == 0 and result.optimizer_updates == 4
    assert [loss.step for loss in result.validation_losses] == [0, 2, 4]
    assert {path: sha256_file(path) for path in before} == before
    assert sha256_file(native.initial_adapter / "adapter_model.safetensors") == (
        identity.initialization.adapter_sha256
    )


@pytest.mark.parametrize(
    "change", ["base", "weights", "alpha", "rank", "overlap", "resume"]
)
def test_initial_adapter_rejects_incompatible_source(
    config: SftExperimentConfig,
    prepared: Path,
    trained: SftTrainingReport,
    tmp_path: Path,
    change: str,
) -> None:
    source = trained.final_adapter
    destination = tmp_path / "initial-destination"
    if change == "base":
        (Path(config.model.name_or_path) / "model.safetensors").write_bytes(b"other")
    elif change == "weights":
        (source / "adapter_model.safetensors").write_bytes(b"changed")
    elif change in ("alpha", "rank"):
        config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "lora": config.training.lora.model_copy(update={change: 10})
                    }
                )
            }
        )
    elif change == "overlap":
        destination = source.parent
    else:
        with pytest.raises(ValueError, match="mutually exclusive"):
            train.train(
                config, destination, init_adapter=source, resume_from=source.parent
            )
        return
    with pytest.raises((RuntimeError, ValueError)):
        train.initialization_source(
            config, destination, Path(config.model.name_or_path), source
        )
    assert not (destination / "training").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"r": LORA_RANK + 1},
        {"lora_alpha": LORA_ALPHA * 2},
        {"lora_dropout": 0.5},
        {"target_modules": ["k_proj"]},
        {"use_rslora": True},
        {"bias": "all"},
    ],
)
def test_cached_adapter_rejects_changed_lora_settings(
    config: SftExperimentConfig,
    prepared: Path,
    trained: SftTrainingReport,
    changes: dict[str, JsonValue],
) -> None:
    path = trained.final_adapter / "adapter_config.json"
    settings = AdapterConfig.model_validate_json(path.read_bytes()).model_dump(
        mode="json"
    )
    settings.update(changes)
    write_json(path, settings)
    with pytest.raises(ValueError, match="recorded LoRA settings"):
        train.checkpoint_model(
            config.model, prepared / "training", trained.checkpoints[-1]
        )
