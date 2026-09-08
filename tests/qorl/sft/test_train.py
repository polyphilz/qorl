"""SFT configuration translation, fixed run inputs and explicit checkpoint handoff."""

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import pytest
import torch
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import CosineSchedulerConfig
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
    TrainingIdentity,
    TrainingRow,
)
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


def test_train_records_counts_and_exports_recorded_scale(
    config: SftExperimentConfig,
    prepared: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "native-checkpoint"
    CHECKPOINT_SAVER.save(
        {
            "app": {
                "model": {
                    "layer.q_proj.lora_A.0": torch.ones(LORA_RANK, CONTEXT_LENGTH),
                    "layer.q_proj.lora_B.0": torch.ones(CONTEXT_LENGTH, LORA_RANK),
                }
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
    result = train.train(config, prepared)
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
        train.checkpoint_model(config.model, prepared / "training", checkpoint)
    shard = next(result.checkpoints[-1].glob("*.distcp"))
    with shard.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="different checkpoint weights"):
        train.checkpoint_model(
            config.model, prepared / "training", result.checkpoints[-1]
        )
