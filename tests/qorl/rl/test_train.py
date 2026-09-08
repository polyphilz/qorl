"""The complete native RL configuration and its explicit run/checkpoint boundaries."""

from pathlib import Path
from typing import Protocol
from unittest.mock import Mock

import pytest
import torch
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import AdamWConfig, CosineSchedulerConfig
from torch.distributed.checkpoint.metadata import Metadata

from qorl.experiment.schemas import RlExperimentConfig, load_config
from qorl.model.files import model_weights_sha256
from qorl.rl import train
from qorl.rl.environment import QorlEnvironment, QorlEnvironmentConfig
from qorl.rl.harness import QorlHarnessConfig
from qorl.rl.schemas import RlTrainingIdentity
from qorl.rl.tasks import QorlTasksetConfig
from qorl.util.hashing import sha256_json
from qorl.util.io import write_json


@pytest.fixture
def config(tmp_path: Path, repository_root: Path) -> RlExperimentConfig:
    loaded = load_config(repository_root / "configs/defaults/000-rl.toml")
    assert isinstance(loaded, RlExperimentConfig)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"test base")
    document = loaded.model_dump()
    document["model"].update(name_or_path=str(base), revision=None)
    document["training"]["renderer"] = {"name": "qwen3.5"}
    document["postgres"]["path"] = (
        repository_root / "docker/postgres/configs/000-pgconf-default"
    )
    document["pool"]["path"] = (
        repository_root / "docker/worker_pool/configs/002-poolconf-4x8"
    )
    document["resources"] = {"training_gpu_ids": [3, 5], "serving_gpu_ids": [7]}
    return RlExperimentConfig.model_validate(document)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    (path / "training-tasks.json").write_text(
        '{"benchmark_id":"job","task_ids":["job-01a","job-01b"]}'
    )
    return path


@pytest.mark.parametrize("max_grad_norm", [None, 0.0, 0.25, 1.0])
@pytest.mark.parametrize("seed", [42, 91])
def test_complete_translation(
    config: RlExperimentConfig, run: Path, max_grad_norm: float | None, seed: int
) -> None:
    config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(update={"seed": seed}),
            "training": config.training.model_copy(
                update={"max_grad_norm": max_grad_norm}
            ),
        }
    )
    native = train.native_config(config, run)
    native = RLConfig.model_validate_json(native.model_dump_json())
    trainer, orchestrator = native.trainer, native.orchestrator
    assert trainer.seed == config.experiment.seed
    source = orchestrator.train.source[0]
    assert trainer.optim.max_norm == max_grad_norm
    assert "max_norm" in trainer.optim.model_dump()
    assert "scheduler" not in trainer.optim.model_dump()
    assert trainer.optim.lr == 1e-6 and trainer.optim.weight_decay == 0
    assert isinstance(trainer.optim, AdamWConfig)
    assert (trainer.optim.betas1, trainer.optim.betas2) == (0.9, 0.999)
    assert trainer.scheduler.type == "constant"
    assert trainer.model.lora is not None
    assert (
        trainer.model.lora.model_dump(exclude={"modules_to_save"})
        == config.training.lora.model_dump()
    )
    assert trainer.model.compile is None
    assert trainer.model.seq_len == orchestrator.seq_len == config.model.context_length
    assert trainer.model.impl == "custom" and trainer.model.attn == "flash_attention_2"
    assert trainer.max_steps == orchestrator.max_steps == 100
    assert trainer.output_dir == orchestrator.output_dir == run / "training"
    assert (
        trainer.weight_broadcast.type
        == orchestrator.weight_broadcast.type
        == "filesystem"
    )
    assert native.resume is trainer.resume is orchestrator.resume is None
    assert orchestrator.eval is None
    assert not orchestrator.constant_trainer_batch_size
    assert (orchestrator.batch_size, source.group_size, orchestrator.group_size) == (
        16,
        4,
        4,
    )
    assert orchestrator.max_off_policy_steps == 2
    assert orchestrator.concurrency.max_inflight == 4
    assert source.serve.max_concurrent == 4
    assert source.serve.pool.type == "static" and source.serve.pool.num_workers == 1
    assert orchestrator.num_train_workers == 2
    assert train.gpu_ids(config) == [7, 3, 5]
    assert source.algo is not None and source.algo.type == "qorl_anchored_grpo"
    assert source.algo.expected_group_size == 4
    assert (
        source.algo.tau,
        source.algo.c,
        source.algo.d,
        source.algo.t,
        source.algo.min_peers,
    ) == (0.05, 0.1, 0.02, 0.1, 2)
    assert orchestrator.renderer.name == "qwen3.5"
    assert source.sampling.top_k == 20 and source.sampling.max_completion_tokens == 2048
    assert native.inference is not None and native.inference.enable_return_sampling_mask
    assert native.inference.router is None
    assert native.inference.vllm.enable_lora
    assert native.inference.vllm.max_lora_rank == 16
    assert native.inference.vllm.reasoning_parser == "qwen3"
    assert (
        native.inference.vllm.lora_target_modules == config.training.lora.target_modules
    )
    assert native.inference.vllm.max_model_len == config.model.context_length
    assert trainer.ckpt is not None and orchestrator.ckpt is not None
    assert (
        trainer.ckpt.interval,
        trainer.ckpt.keep_last,
        trainer.ckpt.keep_interval,
    ) == (5, 2, 10)
    assert trainer.ckpt.skip_progress and trainer.ckpt.skip_optimizer
    assert orchestrator.ckpt.skip_progress
    assert isinstance(source.env, QorlEnvironmentConfig)
    environment = QorlEnvironment(source.env)
    assert isinstance(environment.config.taskset, QorlTasksetConfig)
    harness = source.env.agent.harness
    assert isinstance(harness, QorlHarnessConfig)
    assert harness.agent == config.agent
    assert harness.model is not None and harness.model.max_concurrent_requests == 8
    assert source.env.timeout.episode is source.env.agent.timeout.rollout is None


def test_explicit_harness_settings_survive_construction(
    config: RlExperimentConfig, run: Path
) -> None:
    doc = config.model_dump()
    doc["agent"]["maximum_model_turns"] = 7
    doc["measurement"]["default_timeout_seconds"] = 42.0
    doc["measurement"]["paired_measurements"] = 2
    doc["model"]["max_concurrent_requests"] = 2
    changed = RlExperimentConfig.model_validate(doc)
    native = train.native_config(changed, run)
    source = native.orchestrator.train.source[0]
    assert isinstance(source.env, QorlEnvironmentConfig)
    env = QorlEnvironment(source.env)
    assert env.config.agent.harness is not None
    actual = QorlHarnessConfig.model_validate(env.config.agent.harness.model_dump())
    assert actual.agent == changed.agent
    assert actual.measurement == changed.measurement
    assert actual.model is not None and actual.model.max_concurrent_requests == 2


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("candidate_attempts", 2, "candidate_attempts"),
        ("min_p", 0.1, "sampling replay"),
        ("presence_penalty", 0.5, "sampling replay"),
        ("repetition_penalty", 1.1, "sampling replay"),
        ("temperature", 0.0, "positive"),
        ("top_k", 513, "top_k"),
    ],
)
def test_reject_incompatible_workflows(
    config: RlExperimentConfig, run: Path, field: str, value: float, match: str
) -> None:
    doc = config.model_dump()
    doc["agent" if field == "candidate_attempts" else "inference"][field] = value
    with pytest.raises(ValueError, match=match):
        train.native_config(RlExperimentConfig.model_validate(doc), run)
    assert not (run / "training").exists()


def test_auto_renderer_does_not_guess_local_identity(
    config: RlExperimentConfig, run: Path
) -> None:
    doc = config.model_dump()
    del doc["training"]["renderer"]
    with pytest.raises(ValueError, match=r"training\.renderer"):
        train.native_config(RlExperimentConfig.model_validate(doc), run)


def test_omitted_clipping_is_disabled(config: RlExperimentConfig, run: Path) -> None:
    doc = config.model_dump()
    del doc["training"]["max_grad_norm"]
    native = train.native_config(RlExperimentConfig.model_validate(doc), run)
    assert native.trainer.optim.max_norm is None


def test_scheduler_checked_by_complete_config(
    config: RlExperimentConfig, run: Path
) -> None:
    optimizer = config.training.optimizer.model_copy(
        update={"scheduler": CosineSchedulerConfig(warmup_steps=101, min_lr=0.0)}
    )
    changed = config.model_copy(
        update={"training": config.training.model_copy(update={"optimizer": optimizer})}
    )
    with pytest.raises(ValueError):
        train.native_config(changed, run)


def test_retention_keeps_final_checkpoint(
    config: RlExperimentConfig, run: Path
) -> None:
    doc = config.model_dump()
    doc["training"].update(
        max_steps=4, checkpoints={"type": "interval", "interval": 1, "keep_interval": 3}
    )
    native = train.native_config(RlExperimentConfig.model_validate(doc), run)
    assert native.trainer.ckpt is not None and native.orchestrator.ckpt is not None
    assert native.trainer.ckpt.keep_last == native.orchestrator.ckpt.keep_last == 1


def test_no_restart_over_partial_outputs(
    config: RlExperimentConfig, run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "training").mkdir()
    (run / "training/partial").write_text("preserve")
    launch = Mock()
    monkeypatch.setattr(train, "launch", launch)
    with pytest.raises(ValueError, match="training output exists"):
        train.train(config, run)
    launch.assert_not_called()
    assert (run / "training/partial").read_text() == "preserve"


def record_training(config: RlExperimentConfig, run: Path) -> Path:
    native = train.native_config(config, run)
    training = run / "training"
    write_json(training / train.NATIVE_CONFIG, native.model_dump(mode="json"))
    write_json(
        training / train.TRAINING_IDENTITY,
        RlTrainingIdentity(
            experiment_sha256=sha256_json(config.model_dump(mode="json")),
            base_weights_sha256=model_weights_sha256(Path(native.trainer.model.name)),
            trainer_source="test",
            trainer_config_sha256=sha256_json(native.trainer.model_dump(mode="json")),
        ).model_dump(mode="json"),
    )
    return training


def test_explicit_native_checkpoint_required(
    config: RlExperimentConfig, run: Path
) -> None:
    training = record_training(config, run)
    with pytest.raises(ValueError, match="complete native checkpoint"):
        train.checkpoint_model(config.model, training, run / "arbitrary-adapter")


@pytest.fixture
def checkpoint(config: RlExperimentConfig, run: Path) -> Path:

    import torch
    from torch.distributed.checkpoint import state_dict_saver

    saver: Saver = state_dict_saver
    training = record_training(config, run)
    path = training / "checkpoints/step_4/trainer"
    save_checkpoint(
        saver,
        path,
        {
            "app": {
                "model": {
                    "base_model.model.layers.0.q_proj.lora_A.0": torch.ones((16, 2)),
                    "base_model.model.layers.0.q_proj.lora_B.0": torch.ones((2, 16)),
                }
            }
        },
    )
    return path


def test_checkpoint_serves_recorded_base_and_scale(
    config: RlExperimentConfig, run: Path, checkpoint: Path
) -> None:
    model = train.checkpoint_model(config.model, run / "training", checkpoint.parent)
    assert model.name_or_path == config.model.name_or_path
    assert model.adapter_path == checkpoint.parent / "adapter"
    assert model.revision is None
    assert train.checkpoint_model(config.model, run / "training", checkpoint) == model


@pytest.mark.parametrize(
    "mutation",
    ["tensors", "alpha", "rank", "dropout", "targets", "rslora", "base", "checkpoint"],
)
def test_changed_checkpoint_export_is_rejected(
    config: RlExperimentConfig, run: Path, checkpoint: Path, mutation: str
) -> None:
    import json

    model = train.checkpoint_model(config.model, run / "training", checkpoint)
    assert model.adapter_path is not None
    adapter = model.adapter_path
    if mutation == "tensors":
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    elif mutation == "base":
        (Path(config.model.name_or_path) / "model.safetensors").write_bytes(b"changed")
    elif mutation == "checkpoint":
        with next(checkpoint.glob("*.distcp")).open("ab") as stream:
            stream.write(b"changed")
    else:
        path = adapter / "adapter_config.json"
        doc = json.loads(path.read_bytes())
        field, value = {
            "alpha": ("lora_alpha", 64),
            "rank": ("r", 8),
            "dropout": ("lora_dropout", 0.1),
            "targets": ("target_modules", ["k_proj"]),
            "rslora": ("use_rslora", True),
        }[mutation]
        doc[field] = value
        path.write_text(json.dumps(doc))
    with pytest.raises((ValueError, RuntimeError), match=r"changed|different|LoRA"):
        train.checkpoint_model(config.model, run / "training", checkpoint)


class Saver(Protocol):
    def save(
        self,
        state_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
        *,
        checkpoint_id: Path,
    ) -> Metadata: ...


def save_checkpoint(
    saver: Saver, path: Path, tensors: dict[str, dict[str, dict[str, torch.Tensor]]]
) -> None:
    saver.save(tensors, checkpoint_id=path)


@pytest.mark.parametrize("returncode", [0, 2])
def test_native_launch_owns_only_its_process_and_env(
    config: RlExperimentConfig,
    run: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    import asyncio
    import sys

    from verifiers.v1.serve import EnvServer

    native = train.native_config(config, run)
    servers: list[EnvServer] = []
    commands: list[tuple[str, ...]] = []
    environments: list[dict[str, str]] = []
    spawn = asyncio.create_subprocess_exec

    async def serve(server: EnvServer) -> None:
        servers.append(server)
        await asyncio.Event().wait()

    async def start(
        *command: str, cwd: Path, env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        commands.append(command)
        environments.append(env)
        return await spawn(
            sys.executable, "-c", f"raise SystemExit({returncode})", cwd=cwd, env=env
        )

    monkeypatch.setattr(EnvServer, "run", serve)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    if returncode:
        with pytest.raises(RuntimeError, match="exited with status 2"):
            asyncio.run(train.launch(native, train.gpu_ids(config)))
    else:
        asyncio.run(train.launch(native, train.gpu_ids(config)))
    assert commands[0][1:4] == ("-m", "prime_rl.entrypoints.rl", "@")
    assert environments[0]["CUDA_VISIBLE_DEVICES"] == "7,3,5"
    recorded = RLConfig.model_validate_json(
        (run / "training" / train.NATIVE_CONFIG).read_bytes()
    )
    assert recorded.orchestrator.train.source[0].serve.address == servers[0].address
    assert not servers[0].address.endswith(":0")
    assert servers[0].frontend.closed and servers[0].ctx.closed
