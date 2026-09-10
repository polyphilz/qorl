"""Translate prepared SFT inputs and launch Prime-RL's ordinary training loop."""

import math
import os
import shutil
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
from prime_rl.configs.shared import ResumeConfig, RunConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    CompileConfig,
    LoRAConfig,
    ModelConfig,
)
from safetensors.torch import load as load_tensors
from torch.distributed.checkpoint.filesystem import FileSystemReader

from qorl.adapters.config import adapter_config
from qorl.adapters.export import (
    checkpoint_sha256,
    export_adapter,
    export_configuration,
)
from qorl.adapters.schemas import AdapterExportManifest, LoraSettings
from qorl.adapters.verify import verify_adapter_base, verify_adapter_weights
from qorl.experiment.schemas import SftExperimentConfig, load_config
from qorl.model.files import model_weights_sha256, resolve_model
from qorl.model.schemas import ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.sft.dataset import require_both_splits, source_file
from qorl.sft.schemas import (
    DatasetPreparationIdentity,
    DatasetPreparationReport,
    SftContinuation,
    SftInitialization,
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
    keep_last = settings.checkpoints.keep_last
    if keep_last is None and settings.checkpoints.keep_interval is not None:
        # Prime-RL otherwise deletes a final step outside the retained intervals.
        keep_last = 1
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
            keep_last=keep_last,
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
    """Export selected native weights with a verified local copy of their saved base."""
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
    base = resolve_model(model)
    if model_weights_sha256(base) != identity.base_weights_sha256:
        raise ValueError("training base weights changed")
    lora = native.model.lora
    if lora is None:
        raise ValueError("checkpoint evaluation requires recorded LoRA settings")
    settings = LoraSettings(
        rank=lora.rank,
        alpha=lora.alpha,
        dropout=lora.dropout,
        target_modules=lora.target_modules,
    )
    adapter = checkpoint.parent / "adapter"
    if adapter.exists():
        verify_adapter_base(adapter, base)
        exported = AdapterExportManifest.model_validate_json(
            (adapter / "qorl-manifest.json").read_bytes()
        )
        if exported.checkpoint_sha256 != checkpoint_sha256(checkpoint):
            raise ValueError("exported adapter belongs to different checkpoint weights")
        weights = adapter / "adapter_model.safetensors"
        if sha256_file(weights) != exported.adapter_sha256:
            raise ValueError("exported adapter weights changed")
        expected = export_configuration(
            base, settings, load_tensors(weights.read_bytes())
        )
        if adapter_config(adapter).model_dump(
            exclude={"base_model_name_or_path"}
        ) != expected.model_dump(exclude={"base_model_name_or_path"}):
            raise ValueError("exported adapter differs from recorded LoRA settings")
    else:
        export_adapter(checkpoint, base, settings, adapter)
    return model.model_copy(
        update={"name_or_path": str(base), "revision": None, "adapter_path": adapter}
    )


def continuation_source(
    config: SftExperimentConfig, run: Path, native: SFTConfig, checkpoint: Path
) -> SftContinuation:
    """Verify a completed run without writing or exporting anything in its tree."""
    run = run.resolve()
    checkpoint = checkpoint.expanduser().resolve()
    source_training = checkpoint.parent.parent
    source_run = source_training.parent
    if run.is_relative_to(source_run) or source_run.is_relative_to(run):
        raise ValueError("continuation source and destination runs must not overlap")
    if source_training.name != "training" or checkpoint.parent.name != "checkpoints":
        raise ValueError("--resume-from requires a QORL native step directory")
    source_config = load_config(source_run / "config.toml")
    if not isinstance(source_config, SftExperimentConfig):
        raise ValueError("continuation requires an SFT source")
    source_report = SftTrainingReport.model_validate_json(
        (source_training / "report.json").read_bytes()
    )
    source_identity = TrainingIdentity.model_validate_json(
        (source_training / TRAINING_IDENTITY).read_bytes()
    )
    source_native = SFTConfig.model_validate_json(
        (source_training / TRAINER_CONFIG).read_bytes()
    )
    if source_identity.experiment_sha256 != sha256_json(
        source_config.model_dump(mode="json")
    ):
        raise ValueError("source experiment configuration changed")
    if source_identity.trainer_config_sha256 != sha256_json(
        source_native.model_dump(mode="json")
    ):
        raise ValueError("source trainer configuration changed")
    source_prepared = prepared_report(source_config, source_run / "dataset")
    if source_identity.preparation_sha256 != sha256_file(
        source_run / "dataset/report.json"
    ):
        raise ValueError("source preparation report changed")
    for split in ("training", "validation"):
        relative = Path(split) / "packed.jsonl"
        if sha256_file(source_run / "dataset" / relative) != sha256_file(
            run / "dataset" / relative
        ):
            raise ValueError(f"continuation {split} packed rows differ from source")
    source_preparation = DatasetPreparationIdentity.model_validate_json(
        (source_run / "dataset/preparation.json").read_bytes()
    )
    preparation = DatasetPreparationIdentity.model_validate_json(
        (run / "dataset/preparation.json").read_bytes()
    )
    if (source_preparation.tokenizer_sha256, source_preparation.renderer) != (
        preparation.tokenizer_sha256,
        preparation.renderer,
    ):
        raise ValueError("continuation tokenizer or renderer identity differs")
    completed = source_report.optimizer_updates
    steps_per_epoch = math.ceil(
        source_prepared.training.packed_rows / source_config.training.batch_size
    )
    if (
        completed != source_config.training.epochs * steps_per_epoch
        or source_report.steps_per_epoch != steps_per_epoch
        or source_report.epochs != source_config.training.epochs
        or source_report.training != source_prepared.training
        or source_report.validation != source_prepared.validation
        or source_native.max_steps != completed
        or checkpoint.name != f"step_{completed}"
        or checkpoint / "trainer" not in checkpoint_paths(source_training)
        or not source_report.validation_losses
        or source_report.validation_losses[-1].step != completed
        or source_report.final_adapter
        != source_native.run_dir / "checkpoints" / checkpoint.name / "adapter"
    ):
        raise ValueError(
            "continuation requires the completed run's final epoch checkpoint"
        )
    if config.training.epochs <= source_config.training.epochs:
        raise ValueError("continuation requires additional total epochs")
    if native.scheduler.type != "constant" or config.training.lora.dropout != 0:
        raise ValueError(
            "continuation requires constant scheduling and zero LoRA dropout"
        )
    if source_native.data.type != "prepared" or native.data.type != "prepared":
        raise ValueError("continuation requires prepared data")
    if (
        source_native.model.model_dump(exclude={"name"})
        != native.model.model_dump(exclude={"name"})
        or source_native.optim != native.optim
        or source_native.scheduler != native.scheduler
        or source_native.renderer != native.renderer
        or source_native.tokenizer.model_dump(exclude={"name"})
        != native.tokenizer.model_dump(exclude={"name"})
        or source_native.deployment != native.deployment
        or source_native.matmul_precision != native.matmul_precision
        or (
            source_native.val.model_dump(exclude={"data": {"path"}})
            if source_native.val
            else None
        )
        != (native.val.model_dump(exclude={"data": {"path"}}) if native.val else None)
        or source_native.data.model_dump(exclude={"path", "epochs"})
        != native.data.model_dump(exclude={"path", "epochs"})
    ):
        raise ValueError("continuation training parameters differ from source")
    if (
        model_weights_sha256(Path(native.model.name))
        != source_identity.base_weights_sha256
    ):
        raise ValueError("continuation base weights differ from source")
    checksum = checkpoint_sha256(checkpoint / "trainer")
    keys = FileSystemReader(checkpoint / "trainer").read_metadata().state_dict_metadata
    if not {"app.progress.step", "app.scheduler.last_epoch"} <= keys.keys() or any(
        not any(
            key.startswith("app.optimizers.state.") and key.endswith(suffix)
            for key in keys
        )
        for suffix in (".step", ".exp_avg", ".exp_avg_sq")
    ):
        raise ValueError(
            "source checkpoint lacks AdamW, scheduler or completed progress state"
        )
    adapter = checkpoint / "adapter"
    exported = AdapterExportManifest.model_validate_json(
        (adapter / "qorl-manifest.json").read_bytes()
    )
    if exported.checkpoint_sha256 != checksum:
        raise ValueError("source checkpoint differs from its exported identity")
    if sha256_file(adapter / "adapter_model.safetensors") != exported.adapter_sha256:
        raise ValueError("source adapter weights changed")
    verify_adapter_base(adapter, Path(native.model.name))
    expected = export_configuration(
        Path(native.model.name),
        source_config.training.lora,
        load_tensors((adapter / "adapter_model.safetensors").read_bytes()),
    )
    if adapter_config(adapter).model_dump(
        exclude={"base_model_name_or_path"}
    ) != expected.model_dump(exclude={"base_model_name_or_path"}):
        raise ValueError("source adapter LoRA settings changed")
    return SftContinuation(
        checkpoint=checkpoint, checkpoint_sha256=checksum, completed_steps=completed
    )


def initialization_source(
    config: SftExperimentConfig,
    run: Path,
    base: Path,
    adapter: Path,
) -> SftInitialization:
    """Verify an exported adapter without depending on its original dataset or optimizer."""
    adapter = adapter.expanduser().resolve()
    if adapter.is_relative_to(run.resolve()) or run.resolve().is_relative_to(adapter):
        raise ValueError("initial adapter and destination run must not overlap")
    manifest = verify_adapter_weights(adapter)
    verify_adapter_base(adapter, base)
    tensors = load_tensors((adapter / "adapter_model.safetensors").read_bytes())
    expected = export_configuration(base, config.training.lora, tensors)
    if adapter_config(adapter).model_dump(
        exclude={"base_model_name_or_path"}
    ) != expected.model_dump(exclude={"base_model_name_or_path"}):
        raise ValueError("initial adapter differs from the experiment's LoRA settings")
    return SftInitialization(
        adapter=adapter,
        adapter_sha256=manifest.adapter_sha256,
        config_sha256=sha256_file(adapter / "adapter_config.json"),
        manifest_sha256=sha256_file(adapter / "qorl-manifest.json"),
    )


def train(
    config: SftExperimentConfig,
    run: Path,
    *,
    resume_from: Path | None = None,
    init_adapter: Path | None = None,
) -> SftTrainingReport:
    """Run a fixed prepared schedule without replacing an existing training attempt."""
    if resume_from is not None and init_adapter is not None:
        raise ValueError("--init-adapter and --resume-from are mutually exclusive")
    run = run.resolve()
    training = run / "training"
    if (training / "report.json").exists():
        raise ValueError("SFT training is already complete")
    if training.exists():
        raise ValueError("training output exists; start a new run")
    report = prepared_report(config, run / "dataset")
    native = trainer_config(config, run, report)
    continuation = (
        continuation_source(config, run, native, resume_from)
        if resume_from is not None
        else None
    )
    starting_step = continuation.completed_steps if continuation else 0
    if continuation:
        native.resume = ResumeConfig(step=None, dir=continuation.checkpoint)
    initialization = (
        initialization_source(config, run, Path(native.model.name), init_adapter)
        if init_adapter is not None
        else None
    )
    if initialization:
        snapshot = training / "configs/initial_adapter"
        snapshot.mkdir(parents=True)
        for name, checksum in (
            ("adapter_model.safetensors", initialization.adapter_sha256),
            ("adapter_config.json", initialization.config_sha256),
            ("qorl-manifest.json", initialization.manifest_sha256),
        ):
            shutil.copyfile(initialization.adapter / name, snapshot / name)
            if sha256_file(snapshot / name) != checksum:
                raise ValueError(f"initial adapter changed while copying: {name}")
        native.initial_adapter = snapshot
    identity = TrainingIdentity(
        experiment_sha256=sha256_json(config.model_dump(mode="json")),
        preparation_sha256=sha256_file(run / "dataset/report.json"),
        base_weights_sha256=model_weights_sha256(Path(native.model.name)),
        trainer_source=distribution("prime-rl").read_text("direct_url.json") or "",
        trainer_config_sha256=sha256_json(native.model_dump(mode="json")),
        continuation=continuation,
        initialization=initialization,
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
    if set(losses) != set(range(starting_step, updates + 1, steps_per_epoch)):
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
        starting_step=starting_step,
        validation_losses=[losses[step] for step in sorted(losses)],
        checkpoints=checkpoints,
        final_adapter=evaluated_model.adapter_path,
    )
    write_json(training / "report.json", result.model_dump(mode="json"))
    return result
