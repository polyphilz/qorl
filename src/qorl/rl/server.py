"""An externally owned native EnvServer with a small QORL control exchange."""

import asyncio
import fcntl
import logging
import signal
import socket
import subprocess
from collections.abc import Awaitable, Sequence
from contextlib import suppress
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import msgpack
import renderers
from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import check_inference_ready
from renderers import Renderer, RendererConfig
from renderers.base import load_tokenizer
from transformers import PreTrainedTokenizerFast
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.serve import EnvClient, EnvServer
from verifiers.v1.serve.types import RunRequest, RunResponse

from qorl.experiment.schemas import RlExperimentConfig, load_config
from qorl.model.schemas import JsonValue, LocalInferenceSettings
from qorl.paths import REPOSITORY_ROOT
from qorl.postgres.config import PostgresConfig
from qorl.rl import runtime
from qorl.rl.evidence import write_native
from qorl.rl.schemas import (
    EnvironmentClaim,
    EnvironmentContract,
    EnvironmentControlRequest,
    EnvironmentControlResponse,
    EnvironmentServiceIdentity,
    QorlEnvironmentConfig,
    QorlHarnessConfig,
    QorlTaskData,
    RemoteEnvironmentSettings,
)
from qorl.taskset.schemas import TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.hashing import sha256_file, sha256_json
from qorl.util.io import write_json
from qorl.worker_pool.config import load_pool_config

CONTROL_TIMEOUT = 10.0
HEALTH_INTERVAL = 2.0
HEALTH_TIMEOUT_LIMIT = 3
logger = logging.getLogger(__name__)


class ControlCodec(Protocol):
    def packb(self, o: JsonValue, /, *, use_bin_type: bool) -> bytes | None: ...
    def unpackb(self, packed: bytes, /, *, raw: bool) -> object: ...


class ControlSocket(Protocol):
    def send_multipart(self, frames: Sequence[bytes], /) -> Awaitable[object]: ...


class RendererFactory(Protocol):
    def create_renderer(
        self, tokenizer: PreTrainedTokenizerFast, config: RendererConfig
    ) -> Renderer: ...


CODEC: ControlCodec = msgpack
RENDERERS: RendererFactory = renderers


def validate_renderer(assets: Path, renderer: RendererConfig) -> None:
    tokenizer = load_tokenizer(str(assets))
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("QORL remote rendering requires a fast tokenizer")
    RENDERERS.create_renderer(tokenizer, renderer)


def code_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def contract(
    config: RlExperimentConfig, inputs: Path, assets: Path
) -> EnvironmentContract:
    from qorl.rl.train import renderer_config

    if (
        not (assets / "config.json").is_file()
        or not (assets / "tokenizer.json").is_file()
    ):
        raise ValueError(
            "remote rendering requires local config.json and tokenizer.json assets"
        )
    selected = renderer_config(config)
    if selected.name != "qwen3.5":
        raise ValueError("remote RL requires the explicit qwen3.5 renderer")
    selection_path = inputs / config.data.training.file
    selection = TaskSelection.model_validate_json(selection_path.read_bytes())
    task_set = TaskSet.load(REPOSITORY_ROOT, selection.benchmark_id.value)
    tasks = task_set.resolve(selection)
    for task in tasks:
        task_set.load_sql(task)
    postgres = PostgresConfig.load(REPOSITORY_ROOT / config.postgres.path)
    pool = load_pool_config(REPOSITORY_ROOT / config.pool.path)
    files = {
        file
        for pattern in (
            "config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.*",
            "merges.txt",
            "chat_template*",
            "chat_templates/**/*",
        )
        for file in assets.glob(pattern)
        if file.is_file()
    }
    dependencies = {
        name: version(name)
        for name in (
            "prime-rl",
            "verifiers",
            "renderers",
            "transformers",
            "tokenizers",
        )
    }
    for name in ("prime-rl", "verifiers", "renderers"):
        dependencies[f"{name}-source"] = (
            distribution(name).read_text("direct_url.json") or ""
        )
    return EnvironmentContract(
        experiment_sha256=sha256_json(config.model_dump(mode="json")),
        selection_sha256=sha256_file(selection_path),
        task_sql_sha256={task.task_id: task.sql_sha256 for task in tasks},
        tasks_sha256=sha256_json([task.model_dump(mode="json") for task in tasks]),
        postgres_sha256=postgres.pg_conf_sha256,
        postgres_expected_sha256=postgres.expected_sha256,
        pool_sha256=pool.sha256,
        renderer_assets={
            str(file.relative_to(assets)): sha256_file(file) for file in sorted(files)
        },
        renderer=selected,
        dependencies=dependencies,
        source_sha256=sha256_json(
            {
                str(file.relative_to(REPOSITORY_ROOT)): sha256_file(file)
                for file in sorted((REPOSITORY_ROOT / "src/qorl").rglob("*.py"))
            }
        ),
    )


class QorlEnvClient(EnvClient):
    async def control(
        self, request: EnvironmentControlRequest, *, timeout: float = CONTROL_TIMEOUT
    ) -> EnvironmentControlResponse:
        return await self._request(request, EnvironmentControlResponse, timeout=timeout)

    async def monitor(self, service_id: str) -> None:
        timeouts = 0
        while True:
            try:
                response = await self.control(
                    EnvironmentControlRequest(operation="identity"),
                    timeout=CONTROL_TIMEOUT,
                )
            except TimeoutError as error:
                timeouts += 1
                logger.warning(
                    "Remote environment health timed out (%d/%d): %s",
                    timeouts,
                    HEALTH_TIMEOUT_LIMIT,
                    self.address,
                )
                if timeouts >= HEALTH_TIMEOUT_LIMIT:
                    raise RuntimeError(
                        f"remote environment health timed out {timeouts} consecutive "
                        f"times: {self.address}"
                    ) from error
            else:
                if (
                    response.identity is None
                    or response.identity.service_id != service_id
                ):
                    raise RuntimeError(
                        "remote environment service identity changed during training"
                    )
                timeouts = 0
            await asyncio.sleep(HEALTH_INTERVAL)


class QorlEnvServer(EnvServer):
    """Delegate episodes to Verifiers; validate their source and local renderer first."""

    def __init__(
        self,
        config: QorlEnvironmentConfig,
        *,
        experiment: RlExperimentConfig,
        expected: EnvironmentContract,
        assets: Path,
        output: Path,
        address: str,
    ) -> None:
        super().__init__(
            config, address=address, max_concurrent=experiment.training.max_inflight
        )
        self.experiment = experiment
        self.expected = expected
        self.assets = assets
        self.output = output
        self.service_id = uuid4().hex
        self.code_commit = code_commit(REPOSITORY_ROOT)
        self._identity: EnvironmentServiceIdentity | None = None
        self.claim: EnvironmentClaim | None = None
        self.inference_ready = False
        self.closed_to_work = False
        self.probing = asyncio.Lock()
        self.episodes: set[asyncio.Task[None]] = set()
        self.control_socket: ControlSocket = self.frontend
        if not isinstance(experiment.inference, LocalInferenceSettings):
            raise ValueError("remote RL requires local inference settings")
        self.startup_timeout = experiment.inference.serving.startup_timeout_seconds

    def identity(self) -> EnvironmentServiceIdentity:
        if self._identity is None:
            raise RuntimeError("environment identity is not ready")
        return self._identity

    def capture_identity(self) -> EnvironmentServiceIdentity:
        """Native dispatch starts only after env.serving() has verified PostgreSQL."""
        return EnvironmentServiceIdentity(
            service_id=self.service_id,
            hostname=socket.gethostname(),
            repository=REPOSITORY_ROOT,
            renderer_model=self.assets,
            evidence_directory=self.output,
            address=self.address,
            contract=self.expected,
            code_commit=self.code_commit,
            database_pool=runtime.current().pool_manifest(),
        )

    async def probe(self) -> None:
        async with self.probing:
            if self.inference_ready:
                return
            if self.claim is None:
                raise ValueError("environment service has not been claimed")
            async with asyncio.timeout(self.startup_timeout):
                await check_inference_ready(
                    ClientConfig(
                        base_url=self.experiment.model.base_url or "",
                        wait_for_ready_timeout=self.startup_timeout,
                    ),
                    self.claim.renderer_source,
                )
            self.inference_ready = True

    async def control(
        self, request: EnvironmentControlRequest
    ) -> EnvironmentControlResponse:
        if request.operation == "claim":
            claim = request.claim
            if claim is None or claim.contract != self.expected:
                raise ValueError(
                    "remote environment configuration/task/assets/code identity mismatch"
                )
            if self.closed_to_work or (self.claim is not None and claim != self.claim):
                raise ValueError(
                    "environment service already belongs to another training invocation; restart it"
                )
            if any(
                sha256_file(self.assets / name) != digest
                for name, digest in self.expected.renderer_assets.items()
            ):
                raise ValueError(
                    "environment renderer assets changed since service startup"
                )
            self.claim = claim
            write_json(self.output / "claim.json", claim.model_dump(mode="json"))
        elif request.operation in {"probe", "cancel"}:
            if self.claim is None or request.run_id != self.claim.run_id:
                raise ValueError("remote environment run identity mismatch")
            if request.operation == "probe":
                await self.probe()
            else:
                self.closed_to_work = True
                for work in list(self.episodes):
                    work.cancel()
                # Harnesses preserve partial records and release workers after their threads finish.
                await runtime.drain()
        identity = self.identity()
        if request.operation in {"claim", "cancel"}:
            write_json(self.output / "identity.json", identity.model_dump(mode="json"))
        return EnvironmentControlResponse(
            identity=identity, inference_ready=self.inference_ready
        )

    async def _handle(
        self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes
    ) -> None:
        if self._identity is None:
            self._identity = self.capture_identity()
        if method != EnvironmentControlRequest.method.encode():
            await super()._handle(client_id, request_id, method, payload)
            return
        try:
            response = await self.control(
                EnvironmentControlRequest.model_validate(
                    CODEC.unpackb(payload, raw=False)
                )
            )
        except Exception as error:
            response = EnvironmentControlResponse(success=False, error=str(error))
        encoded = CODEC.packb(response.model_dump(mode="json"), use_bin_type=True)
        if encoded is None:
            raise RuntimeError("control response encoding returned no bytes")
        await self.control_socket.send_multipart(
            [
                client_id,
                request_id,
                encoded,
            ]
        )

    async def _run(self, req: RunRequest) -> RunResponse:
        data = QorlTaskData.model_validate(
            req.model_dump(include={"task_data"})["task_data"]
        )
        if (
            self.closed_to_work
            or self.claim is None
            or data.remote_run != self.claim.run_id
        ):
            raise ValueError("episode does not belong to the claimed environment run")
        if data.task_id not in self.expected.task_sql_sha256:
            raise ValueError("episode task is outside the saved selection")
        active = runtime.current()
        task = next(
            task for task in active.task_set.tasks if task.task_id == data.task_id
        )
        if (
            task.template_id != data.template_id
            or task.sql_sha256 != self.expected.task_sql_sha256[data.task_id]
        ):
            raise ValueError("episode task identity changed")
        client = req.client
        if (
            not isinstance(client, TrainClientConfig)
            or client.base_url != self.experiment.model.base_url
            or client.renderer != self.expected.renderer
            or client.renderer_model_name != self.claim.renderer_source
        ):
            raise ValueError(
                "episode must use the configured native training renderer and upstream"
            )
        work: asyncio.Task[None] | None = asyncio.current_task()
        if work is None:
            raise RuntimeError("episode must run in an asyncio task")
        self.episodes.add(work)
        try:
            await self.probe()
            response = await super()._run(
                req.model_copy(
                    update={
                        "client": client.model_copy(
                            update={"renderer_model_name": str(self.assets)}
                        )
                    }
                )
            )
        finally:
            self.episodes.discard(work)
        if response.episode is not None:
            await asyncio.to_thread(
                write_native,
                self.output / "episodes" / f"{response.episode.id}.msgpack",
                response.episode,
            )
        return response


async def serve_environment(
    experiment: Path, address: str, assets: Path, output: Path
) -> None:
    from qorl.rl.train import environment_config

    RemoteEnvironmentSettings(address=address)
    config = load_config(experiment / "config.toml")
    if not isinstance(config, RlExperimentConfig):
        raise ValueError("qorl rl serve requires an RL experiment")
    if config.rl.environment is None or config.rl.environment.address != address:
        raise ValueError("--bind must match rl.environment.address")
    assets = assets.expanduser().resolve()
    expected = contract(config, experiment, assets)
    # Build the same native renderer without reading any model weight shard.
    await asyncio.to_thread(validate_renderer, assets, expected.renderer)
    output.mkdir(parents=True, exist_ok=False)
    environment = environment_config(config, experiment, config.model)
    if not isinstance(environment.agent.harness, QorlHarnessConfig):
        raise ValueError("QORL service requires its configured harness")
    environment.agent.harness.evidence_directory = output
    server = QorlEnvServer(
        environment,
        experiment=config,
        expected=expected,
        assets=assets,
        output=output,
        address=address,
    )
    log = logging.FileHandler(output / "service.log")
    previous_level = logging.getLogger().level
    logging.getLogger().addHandler(log)
    logging.getLogger().setLevel(logging.INFO)
    serving = asyncio.create_task(server.run())
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, serving.cancel)
    try:
        await serving
    finally:

        async def cleanup() -> None:
            serving.cancel()
            try:
                with suppress(Exception, asyncio.CancelledError):
                    await serving
            finally:
                server.frontend.close()
                server.ctx.term()
                for signum in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(signum)
                logging.getLogger().removeHandler(log)
                logging.getLogger().setLevel(previous_level)
                log.close()

        cleaning = asyncio.create_task(cleanup())
        while not cleaning.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleaning)
        cleaning.result()


def serve(experiment: Path, address: str, assets: Path, output: Path) -> None:
    """Own one service and pool; other training hosts own their own processes."""
    with Path("/tmp/qorl-rl-environment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another QORL environment service is running") from error
        containers = subprocess.run(
            ["docker", "ps", "--format", "{{.Image}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if any(image.startswith("qorl-postgres:") for image in containers):
            raise RuntimeError("a QORL PostgreSQL pool is already running")
        asyncio.run(
            serve_environment(experiment.resolve(), address, assets, output.resolve())
        )
