"""Translate one saved experiment and launch Prime-RL's normal RL machinery."""

import asyncio
import os
import sys
from contextlib import suppress
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import urlsplit

import renderers.base as rendering
import verifiers.v1 as vf
from prime_rl.configs import inference as native_inference
from prime_rl.configs import orchestrator as native_orchestrator
from prime_rl.configs import trainer as native_trainer
from prime_rl.configs.algorithm import GRPOAlgoConfig, QorlAnchoredGRPOAlgoConfig
from prime_rl.configs.monitors import FileMonitorConfig, OrchestratorMonitorsConfig
from prime_rl.configs.rl import RLConfig, SingleNodeDeploymentConfig
from prime_rl.configs.shared import ClientConfig, RunConfig
from pydantic import TypeAdapter
from renderers import AutoRendererConfig, RendererConfig
from safetensors.torch import load as load_tensors
from verifiers.v1.serve import EnvServer

from qorl.adapters.config import adapter_config
from qorl.adapters.export import checkpoint_sha256, export_adapter, export_configuration
from qorl.adapters.schemas import AdapterExportManifest, LoraSettings
from qorl.adapters.verify import verify_adapter_base
from qorl.experiment.schemas import RlExperimentConfig
from qorl.model.files import model_weights_sha256, resolve_model
from qorl.model.schemas import LocalInferenceSettings, ModelSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.config import PostgresConfig
from qorl.rl.environment import QorlEnvironmentConfig
from qorl.rl.harness import QorlHarnessConfig
from qorl.rl.report import write_report
from qorl.rl.schemas import AnchoredGrpoSettings, RlTrainingIdentity
from qorl.rl.tasks import QorlTasksetConfig
from qorl.taskset.schemas import TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json
from qorl.util.seeds import derive_seed
from qorl.worker_pool.config import load_pool_config

NATIVE_CONFIG = Path("configs/qorl.json")
TRAINING_IDENTITY = Path("configs/identity.json")
RENDERER_CONFIG: TypeAdapter[RendererConfig] = TypeAdapter(RendererConfig)


def gpu_ids(config: RlExperimentConfig) -> list[int]:
    """The native launcher assigns inference first, then training, in visible order."""
    resources = config.resources
    if (
        resources is None
        or not resources.serving_gpu_ids
        or not resources.training_gpu_ids
    ):
        raise ValueError("RL requires explicit serving and training GPU IDs")
    return [*resources.serving_gpu_ids, *resources.training_gpu_ids]


def renderer_config(config: RlExperimentConfig) -> RendererConfig:
    selected = config.training.renderer
    if isinstance(selected, AutoRendererConfig):
        name = rendering.MODEL_RENDERER_MAP.get(config.model.name_or_path)
        if name is None:
            raise ValueError(
                "unknown RL renderer; set training.renderer explicitly for this model"
            )
        selected = RENDERER_CONFIG.validate_python(
            {**selected.model_dump(exclude_unset=True), "name": name}
        )
    if not isinstance(config.inference, LocalInferenceSettings):
        raise ValueError("RL requires local inference")
    options = selected.model_dump(exclude_unset=True)
    options["name"] = selected.name
    if "enable_thinking" in selected.template_field_names():
        options["enable_thinking"] = config.inference.thinking
    elif config.inference.thinking:
        raise ValueError("RL renderer does not support the configured thinking switch")
    return RENDERER_CONFIG.validate_python(options)


def native_config(config: RlExperimentConfig, run: Path) -> RLConfig:
    """Validate complete trainer, orchestrator and inference models before any launch."""
    config = RlExperimentConfig.model_validate(config.model_dump())
    inference = config.inference
    if not isinstance(inference, LocalInferenceSettings):
        raise ValueError("RL requires local inference")
    # Native replay supports temperature/top-p/top-k, not these logit transforms.
    if (
        inference.min_p != 0
        or inference.presence_penalty != 0
        or inference.repetition_penalty != 1
    ):
        raise ValueError(
            "RL sampling replay requires min_p=0, presence_penalty=0 and repetition_penalty=1"
        )
    if inference.temperature <= 0:
        raise ValueError("RL requires positive sampling temperature")
    serving = inference.serving
    address = urlsplit(config.model.base_url or "")
    host = "127.0.0.1" if serving.host == "0.0.0.0" else serving.host
    if (address.scheme, address.hostname, address.port, address.path.rstrip("/")) != (
        "http",
        host,
        serving.port,
        "/v1",
    ):
        raise ValueError(
            "model.base_url must address the configured local server's /v1 endpoint"
        )
    if config.model.api_key_env is not None:
        raise ValueError("managed RL inference does not use an API credential")
    if inference.thinking and serving.reasoning_parser is None:
        raise ValueError("thinking requires inference.serving.reasoning_parser")
    devices = gpu_ids(config)
    assert (
        config.resources is not None and config.resources.training_gpu_ids is not None
    )
    assert config.resources.serving_gpu_ids is not None
    training = config.training
    base = resolve_model(config.model)
    selected_renderer = renderer_config(config)
    # Resolve and verify the saved selection, including SQL, without resampling.
    selection_path = (run / config.data.training.file).resolve()
    selection = TaskSelection.model_validate_json(selection_path.read_bytes())
    task_set = TaskSet.load(REPOSITORY_ROOT, selection.benchmark_id.value)
    for task in task_set.resolve(selection):
        task_set.load_sql(task)
    postgres = (REPOSITORY_ROOT / config.postgres.path).resolve()
    pool = (REPOSITORY_ROOT / config.pool.path).resolve()
    PostgresConfig.load(postgres)
    load_pool_config(pool)
    harness = QorlHarnessConfig(
        model=config.model.model_copy(
            update={"name_or_path": str(base), "revision": None}
        ),
        inference=inference,
        agent=config.agent,
        measurement=config.measurement,
        rl=config.rl,
        tool_timeout=600.0,
        seed=config.experiment.seed,
    )
    environment = QorlEnvironmentConfig(
        id="qorl",
        max_concurrent_agents=1,
        postgres_config=postgres,
        pool_config=pool,
        taskset=QorlTasksetConfig(
            id="qorl",
            selection=selection_path,
            shuffle_seed=derive_seed(config.experiment.seed, "rl-task-order"),
        ),
        agent=vf.AgentConfig(harness=harness, runtime=vf.SubprocessConfig()),
    )
    algorithm = config.rl.algorithm
    native_algorithm = (
        QorlAnchoredGRPOAlgoConfig(
            **algorithm.model_dump(), expected_group_size=training.group_size
        )
        if isinstance(algorithm, AnchoredGrpoSettings)
        else GRPOAlgoConfig()
    )
    checkpoint = training.checkpoints
    keep_last = checkpoint.keep_last
    if keep_last is None and checkpoint.keep_interval is not None:
        keep_last = 1
    optimizer = training.optimizer
    runtime = training.runtime
    return RLConfig(
        output_dir=run.resolve(),
        run=RunConfig(name=config.experiment.name, dir="training"),
        dashboard=False,
        deployment=SingleNodeDeploymentConfig(
            num_train_gpus=len(config.resources.training_gpu_ids),
            num_infer_gpus=len(config.resources.serving_gpu_ids),
            gpus_per_node=len(devices),
        ),
        trainer=native_trainer.TrainerConfig(
            seed=config.experiment.seed,
            model=native_trainer.ModelConfig(
                name=str(base),
                seq_len=config.model.context_length,
                impl=runtime.implementation,
                attn=runtime.attention,
                optimization_dtype=runtime.optimization_dtype,
                reduce_dtype=runtime.reduce_dtype,
                compile=native_trainer.CompileConfig() if runtime.compile else None,
                lora=native_trainer.LoRAConfig.model_validate(
                    training.lora.model_dump()
                ),
            ),
            optim=native_trainer.AdamWConfig(
                lr=optimizer.lr,
                weight_decay=optimizer.weight_decay,
                betas1=optimizer.betas1,
                betas2=optimizer.betas2,
                max_norm=training.max_grad_norm,
            ),
            scheduler=optimizer.scheduler,
            max_steps=training.max_steps,
            ckpt=native_trainer.CheckpointConfig(
                interval=checkpoint.interval,
                keep_last=keep_last,
                keep_interval=checkpoint.keep_interval,
                skip_progress=True,
                skip_optimizer=True,
                skip_scheduler=True,
                skip_dataloader=True,
            ),
        ),
        orchestrator=native_orchestrator.OrchestratorConfig(
            tasks_per_minute=None,
            env_server_base_port=5000,
            token_batch_size=None,
            pad_to_multiple_of=1,
            model=native_orchestrator.ModelConfig(
                name=str(base),
                client=ClientConfig(
                    base_url=config.model.base_url or "",
                    wait_for_ready_timeout=serving.startup_timeout_seconds,
                ),
            ),
            renderer=selected_renderer,
            algo=native_algorithm,
            train=native_orchestrator.TrainConfig(
                source=[
                    native_orchestrator.TrainSourceConfig(
                        name="qorl",
                        ratio=1.0,
                        env=environment,
                        serve=vf.ServeConfig(
                            address="tcp://127.0.0.1:0",
                            pool=vf.StaticPoolConfig(num_workers=1),
                            max_concurrent=training.max_inflight,
                        ),
                        sampling=native_orchestrator.TrainSamplingConfig(
                            temperature=inference.temperature,
                            top_p=inference.top_p,
                            top_k=inference.top_k or None,
                            max_completion_tokens=inference.max_tokens,
                            extra_body={
                                "chat_template_kwargs": {
                                    "enable_thinking": inference.thinking
                                }
                            },
                        ),
                        group_size=training.group_size,
                        algo=native_algorithm,
                    )
                ]
            ),
            constant_trainer_batch_size=False,
            batch_size=training.batch_size,
            group_size=training.group_size,
            concurrency=native_orchestrator.ConcurrencyConfig(
                initial_inflight=training.max_inflight,
                min_inflight=training.max_inflight,
                max_inflight=training.max_inflight,
            ),
            seq_len=config.model.context_length,
            max_steps=training.max_steps,
            max_off_policy_steps=training.max_off_policy_steps,
            num_train_workers=len(config.resources.training_gpu_ids),
            ckpt=native_orchestrator.CheckpointConfig(
                wait_for_weights_timeout=None,
                interval=checkpoint.interval,
                keep_last=keep_last,
                keep_interval=checkpoint.keep_interval,
                skip_progress=True,
            ),
            monitors=OrchestratorMonitorsConfig(
                file=FileMonitorConfig(compress=False, float_decimals=None)
            ),
        ),
        inference=native_inference.InferenceConfig(
            router=None,
            server=native_inference.ServerConfig(
                host=serving.host, port=serving.port, liveness_timeout_seconds=30.0
            ),
            vllm=native_inference.VllmConfig.model_validate(
                {
                    "model": str(base),
                    "dtype": serving.dtype,
                    "max_model_len": config.model.context_length,
                    "tool_call_parser": serving.tool_call_parser,
                    "reasoning_parser": serving.reasoning_parser,
                    "max_num_seqs": serving.max_num_seqs,
                    "gpu_memory_utilization": serving.gpu_memory_utilization,
                    "enable_prefix_caching": serving.enable_prefix_caching,
                    "enforce_eager": True,
                    "tensor_parallel_size": len(config.resources.serving_gpu_ids),
                    "lora_target_modules": training.lora.target_modules,
                    "seed": derive_seed(config.experiment.seed, "rl-inference"),
                    "generation_config": "vllm",
                    "language_model_only": True,
                }
            ),
            env_vars={
                "VLLM_ENFORCE_STRICT_TOOL_CALLING": "0",
                "VLLM_USE_FLASHINFER_SAMPLER": "1"
                if serving.use_flashinfer_sampler
                else "0",
            },
        ),
    )


async def launch(native: RLConfig, devices: list[int]) -> None:
    """Host the normal env server while the native launcher owns GPU processes."""
    source = native.orchestrator.train.source[0]
    server = EnvServer(
        source.env,
        address="tcp://127.0.0.1:0",
        max_concurrent=source.serve.max_concurrent,
    )
    training = native.run_dir
    serving: asyncio.Task[None] | None = None
    process: asyncio.subprocess.Process | None = None
    try:
        source.serve.address = server.address
        # Validate the exact config written, including the allocated server address.
        native = RLConfig.model_validate(native.model_dump())
        write_json(training / NATIVE_CONFIG, native.model_dump(mode="json"))
        serving = asyncio.create_task(server.run())
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "prime_rl.entrypoints.rl",
            "@",
            str(training / NATIVE_CONFIG),
            cwd=REPOSITORY_ROOT,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
                "PATH": str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ.get("PATH", ""),
            },
        )
        waiting = asyncio.create_task(process.wait())
        done, _ = await asyncio.wait(
            (serving, waiting), return_when=asyncio.FIRST_COMPLETED
        )
        if serving in done:
            serving.result()
            raise RuntimeError("RL environment server stopped before training finished")
        if waiting.result() != 0:
            raise RuntimeError(
                f"RL trainer exited with status {waiting.result()}; logs: {training / 'logs'}"
            )
    finally:

        async def cleanup() -> None:
            if process is not None and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                await process.wait()
            try:
                if serving is not None:
                    serving.cancel()
                    with suppress(asyncio.CancelledError):
                        await serving
            finally:
                server.frontend.close()
                server.ctx.term()

        cleaning = asyncio.create_task(cleanup())
        while not cleaning.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleaning)
        cleaning.result()


def train(config: RlExperimentConfig, run: Path) -> None:
    """Start from the configured base; preserve every previous training attempt."""
    run = run.resolve()
    training = run / "training"
    if training.exists():
        raise ValueError("training output exists; start a new run")
    native = native_config(config, run)
    training.mkdir()
    identity = RlTrainingIdentity(
        experiment_sha256=sha256_json(config.model_dump(mode="json")),
        base_weights_sha256=model_weights_sha256(Path(native.trainer.model.name)),
        trainer_source=distribution("prime-rl").read_text("direct_url.json") or "",
        trainer_config_sha256=sha256_json(native.trainer.model_dump(mode="json")),
    )
    write_json(training / TRAINING_IDENTITY, identity.model_dump(mode="json"))
    try:
        asyncio.run(launch(native, gpu_ids(config)))
    finally:
        report = write_report(
            training,
            completed=False,
            anchored=isinstance(config.rl.algorithm, AnchoredGrpoSettings),
        )
    if report.optimizer_steps != list(range(1, config.training.max_steps + 1)):
        raise RuntimeError(
            "native trainer did not record all requested optimizer updates"
        )
    if not (
        training
        / "checkpoints"
        / f"step_{config.training.max_steps}"
        / "trainer/.metadata"
    ).is_file():
        raise RuntimeError("native trainer did not retain the final checkpoint")
    write_json(
        training / "report.json",
        report.model_copy(update={"completed": True}).model_dump(mode="json"),
    )


def checkpoint_model(
    model: ModelSettings, training: Path, checkpoint: Path
) -> ModelSettings:
    """Export one explicit native checkpoint with its recorded base and LoRA settings."""
    native = RLConfig.model_validate_json((training / NATIVE_CONFIG).read_bytes())
    identity = RlTrainingIdentity.model_validate_json(
        (training / TRAINING_IDENTITY).read_bytes()
    )
    if (
        sha256_json(native.trainer.model_dump(mode="json"))
        != identity.trainer_config_sha256
    ):
        raise ValueError("recorded trainer configuration changed")
    checkpoint = (REPOSITORY_ROOT / checkpoint.expanduser()).resolve()
    if (checkpoint / "trainer/.metadata").is_file():
        checkpoint /= "trainer"
    if (
        checkpoint.parent.parent != (training / "checkpoints").resolve()
        or not checkpoint.parent.name.startswith("step_")
        or not checkpoint.parent.name.removeprefix("step_").isdecimal()
        or checkpoint.name != "trainer"
        or not (checkpoint / ".metadata").is_file()
    ):
        raise ValueError(
            "checkpoint must be a complete native checkpoint from this run"
        )
    base = Path(native.trainer.model.name)
    if model_weights_sha256(base) != identity.base_weights_sha256:
        raise ValueError("training base weights changed")
    lora = native.trainer.model.lora
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
        if adapter_config(adapter) != expected:
            raise ValueError("exported adapter differs from recorded LoRA settings")
    else:
        export_adapter(checkpoint, base, settings, adapter)
    return model.model_copy(
        update={"name_or_path": str(base), "revision": None, "adapter_path": adapter}
    )
