"""Translate prepared SFT inputs and launch Prime-RL's ordinary training loop."""

import math
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

from prime_rl.configs.sft import (
    PreparedDataConfig,
    SFTConfig,
    SFTValConfig,
    SingleNodeDeploymentConfig,
)
from prime_rl.configs.shared import RunConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    CompileConfig,
    LoRAConfig,
    ModelConfig,
)

from qorl.adapters.export import checkpoint_sha256, export_adapter
from qorl.adapters.schemas import AdapterExportManifest, LoraSettings
from qorl.adapters.verify import verify_adapter_base
from qorl.experiment.schemas import SftExperimentConfig
from qorl.model.files import model_weights_sha256, resolve_model
from qorl.model.schemas import ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.sft.dataset import require_both_splits, source_file
from qorl.sft.schemas import (
    DatasetPreparationIdentity,
    DatasetPreparationReport,
    SftTrainingReport,
    TrainerMetric,
    TrainingIdentity,
    ValidationLoss,
)
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json

TRAINER_CONFIG = Path("configs/qorl.json")
TRAINING_IDENTITY = Path("configs/identity.json")


def training_gpu_ids(config: SftExperimentConfig) -> list[int]:
    """Require the experiment's explicit training devices."""
    if config.resources is None or not config.resources.training_gpu_ids:
        raise ValueError("training requires resources.training_gpu_ids")
    return config.resources.training_gpu_ids


def prepared_report(
    config: SftExperimentConfig,
    dataset: Path,
) -> DatasetPreparationReport:
    """Require completed, unchanged rows prepared for this exact experiment."""
    identity = DatasetPreparationIdentity.model_validate_json(
        (dataset / "preparation.json").read_bytes()
    )
    if identity.config_sha256 != sha256_json(config.model_dump(mode="json")):
        raise ValueError("prepared dataset belongs to different experiment settings")
    report = DatasetPreparationReport.model_validate_json(
        (dataset / "report.json").read_bytes()
    )
    for name, checksum in report.files.items():
        if sha256_file(source_file(dataset, Path(name))) != checksum:
            raise ValueError(f"prepared dataset file changed: {name}")
    require_both_splits(report)
    return report


def trainer_config(
    config: SftExperimentConfig,
    run: Path,
    report: DatasetPreparationReport,
) -> SFTConfig:
    """Validate the full native configuration, including its finite scheduler length."""
    settings = config.training
    runtime = settings.runtime
    optimizer = settings.optimizer
    steps_per_epoch = math.ceil(report.training.packed_rows / settings.batch_size)
    base = resolve_model(config.model)
    return SFTConfig(
        model=ModelConfig(
            name=str(base),
            seq_len=config.model.context_length,
            impl=runtime.implementation,
            attn=runtime.attention,
            optimization_dtype=runtime.optimization_dtype,
            reduce_dtype=runtime.reduce_dtype,
            compile=CompileConfig() if runtime.compile else None,
            lora=LoRAConfig.model_validate(settings.lora.model_dump()),
        ),
        renderer=settings.renderer,
        data=PreparedDataConfig(
            path=run / "dataset/training/packed.jsonl",
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            micro_batch_size=settings.micro_batch_size,
            num_workers=settings.num_workers,
            seq_len=config.model.context_length,
            seed=config.experiment.seed,
        ),
        val=SFTValConfig(
            eval_on_start=True,
            interval=steps_per_epoch,
            data=PreparedDataConfig(
                epochs=1,
                path=run / "dataset/validation/packed.jsonl",
                shuffle=False,
                batch_size=settings.batch_size,
                micro_batch_size=settings.micro_batch_size,
                num_workers=settings.num_workers,
                seq_len=config.model.context_length,
            ),
        ),
        optim=AdamWConfig(
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay,
            betas1=optimizer.betas1,
            betas2=optimizer.betas2,
            max_norm=settings.max_grad_norm,
        ),
        scheduler=optimizer.scheduler,
        ckpt=CheckpointConfig(
            interval=settings.checkpoints.interval,
            keep_last=settings.checkpoints.keep_last,
            keep_interval=settings.checkpoints.keep_interval,
        ),
        max_steps=settings.epochs * steps_per_epoch,
        output_dir=run,
        run=RunConfig(name=config.experiment.name, dir="training"),
        deployment=SingleNodeDeploymentConfig(
            num_train_gpus=len(training_gpu_ids(config)),
            gpus_per_node=len(training_gpu_ids(config)),
            num_infer_gpus=0,
        ),
        dashboard=False,
    )


def checkpoint_paths(training: Path) -> list[Path]:
    """List complete native checkpoint directories in numerical step order."""
    return sorted(
        (
            path.parent
            for path in (training / "checkpoints").glob("step_*/trainer/.metadata")
        ),
        key=lambda path: int(path.parent.name.removeprefix("step_")),
    )


def checkpoint_model(
    model: ModelSettings,
    training: Path,
    checkpoint: Path,
) -> ModelSettings:
    """Export explicitly selected native weights using their saved base and LoRA settings."""
    native = SFTConfig.model_validate_json((training / TRAINER_CONFIG).read_bytes())
    identity = TrainingIdentity.model_validate_json(
        (training / TRAINING_IDENTITY).read_bytes()
    )
    if sha256_json(native.model_dump(mode="json")) != identity.trainer_config_sha256:
        raise ValueError("recorded trainer configuration changed")
    checkpoint = (REPOSITORY_ROOT / checkpoint.expanduser()).resolve()
    if (checkpoint / "trainer/.metadata").is_file():
        checkpoint /= "trainer"
    if checkpoint not in checkpoint_paths(training):
        raise ValueError(
            "checkpoint must be a complete native checkpoint from this run"
        )
    base = Path(native.model.name)
    if model_weights_sha256(base) != identity.base_weights_sha256:
        raise ValueError("training base weights changed")
    lora = native.model.lora
    if lora is None:
        raise ValueError("checkpoint evaluation requires recorded LoRA settings")
    adapter = checkpoint.parent / "adapter"
    if adapter.exists():
        verify_adapter_base(adapter, base)
        exported = AdapterExportManifest.model_validate_json(
            (adapter / "qorl-manifest.json").read_bytes()
        )
        if exported.checkpoint_sha256 != checkpoint_sha256(checkpoint):
            raise ValueError("exported adapter belongs to different checkpoint weights")
    else:
        export_adapter(
            checkpoint,
            base,
            LoraSettings(
                rank=lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                target_modules=lora.target_modules,
            ),
            adapter,
        )
    return model.model_copy(
        update={"name_or_path": str(base), "revision": None, "adapter_path": adapter}
    )


def train(
    config: SftExperimentConfig,
    run: Path,
) -> SftTrainingReport:
    """Run a fixed prepared schedule without replacing an existing training attempt."""
    run = run.resolve()
    training = run / "training"
    if (training / "report.json").exists():
        raise ValueError("SFT training is already complete")
    if training.exists():
        raise ValueError("training output exists; start a new run")
    report = prepared_report(config, run / "dataset")
    native = trainer_config(config, run, report)
    identity = TrainingIdentity(
        experiment_sha256=sha256_json(config.model_dump(mode="json")),
        preparation_sha256=sha256_file(run / "dataset/report.json"),
        base_weights_sha256=model_weights_sha256(Path(native.model.name)),
        trainer_source=distribution("prime-rl").read_text("direct_url.json") or "",
        trainer_config_sha256=sha256_json(native.model_dump(mode="json")),
    )
    write_json(training / TRAINING_IDENTITY, identity.model_dump(mode="json"))
    write_json(training / TRAINER_CONFIG, native.model_dump(mode="json"))
    command = [
        sys.executable,
        "-m",
        "prime_rl.entrypoints.sft",
        "@",
        str(training / TRAINER_CONFIG),
    ]
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, training_gpu_ids(config)))
    }
    try:
        subprocess.run(command, check=True, env=environment, cwd=REPOSITORY_ROOT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"SFT trainer exited with status {error.returncode}; logs: {training / 'logs'}"
        ) from error
    losses: dict[int, ValidationLoss] = {}
    with (training / "monitors/file/metrics.jsonl").open() as stream:
        for line in stream:
            metric = TrainerMetric.model_validate_json(line)
            if metric.validation_loss is not None:
                losses[metric.step] = ValidationLoss(
                    step=metric.step, loss=metric.validation_loss
                )
    steps_per_epoch = math.ceil(
        report.training.packed_rows / config.training.batch_size
    )
    updates = steps_per_epoch * config.training.epochs
    if set(losses) != set(range(0, updates + 1, steps_per_epoch)):
        raise RuntimeError(
            "trainer did not record validation at each required weight step"
        )
    checkpoints = checkpoint_paths(training)
    final = training / "checkpoints" / f"step_{updates}" / "trainer"
    evaluated_model = checkpoint_model(config.model, training, final)
    if evaluated_model.adapter_path is None:
        raise RuntimeError("final checkpoint has no exported adapter")
    result = SftTrainingReport(
        training=report.training,
        validation=report.validation,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        steps_per_epoch=steps_per_epoch,
        optimizer_updates=updates,
        validation_losses=[losses[step] for step in sorted(losses)],
        checkpoints=checkpoints,
        final_adapter=evaluated_model.adapter_path,
    )
    write_json(training / "report.json", result.model_dump(mode="json"))
    return result
