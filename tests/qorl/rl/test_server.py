"""Real native remote transport and renderer, with only HTTP/SQL work simulated."""

import asyncio
import json
import logging
import socket
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
import verifiers.v1 as vf
from aiohttp import web
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.clients import InferenceClient
from renderers import Qwen35RendererConfig
from tests.qorl.sft.test_dataset import BYTE_ALPHABET, SPECIAL_TOKENS, SavedTokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from qorl import cli
from qorl.agent.schemas import AgentTrace
from qorl.experiment.schemas import RlExperimentConfig, load_config
from qorl.model.schemas import JsonObject
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.rl import harness as harness_module
from qorl.rl import runtime as shared_runtime
from qorl.rl import server, train
from qorl.rl.environment import QorlEnvironment
from qorl.rl.runtime import QorlRuntime
from qorl.rl.schemas import (
    EnvironmentClaim,
    EnvironmentCleanup,
    EnvironmentControlRequest,
    EnvironmentControlResponse,
    EnvironmentServiceIdentity,
    QorlEnvironmentConfig,
    QorlHarnessConfig,
    QorlTaskData,
    QorlTasksetConfig,
    RemoteEnvironmentSettings,
    RlRolloutRecord,
)
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig, PoolManifest


class EpisodeClient(Protocol):
    async def run(
        self,
        client: vf.ClientConfig,
        model: str,
        sampling: vf.SamplingConfig,
        task_data: JsonObject,
    ) -> vf.WireEpisode: ...


async def dispatch(
    client: EpisodeClient,
    inference: InferenceClient,
) -> vf.WireEpisode:
    return await client.run(
        client=inference.train_client,
        model=inference.model_name,
        sampling=vf.SamplingConfig(max_tokens=2048, temperature=1),
        task_data=QorlTaskData(
            task_id="job-01a", template_id="job-01", remote_run="one", prompt="job-01a"
        ).model_dump(),
    )


def save_tokenizer(tokenizer: SavedTokenizer, path: Path) -> None:
    tokenizer.save_pretrained(path)


def encode(tokenizer: SavedTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.fixture
def hybrid(
    tmp_path: Path, repository_root: Path
) -> tuple[RlExperimentConfig, Path, Path]:
    assets = tmp_path / "assets"
    tokens = [*SPECIAL_TOKENS, *sorted(BYTE_ALPHABET.alphabet())]
    backend = Tokenizer(
        models.BPE({token: index for index, token in enumerate(tokens)}, [])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=SPECIAL_TOKENS,
        eos_token="<|im_end|>",
    )
    save_tokenizer(tokenizer, assets)
    (assets / "config.json").write_text('{"model_type":"qwen3"}')
    (assets / "model.safetensors").write_bytes(b"never loaded")
    config = load_config(repository_root / "configs/defaults/000-rl.toml")
    assert isinstance(config, RlExperimentConfig)
    data = config.model_dump()
    data["model"].update(
        name_or_path=str(assets),
        revision=None,
        request_timeout_seconds=2,
        retry={
            "max_attempts": 1,
            "initial_delay_seconds": 0.01,
            "maximum_delay_seconds": 0.01,
        },
    )
    data["training"]["renderer"] = {"name": "qwen3.5"}
    data["inference"]["serving"]["startup_timeout_seconds"] = 17
    data["postgres"]["path"] = Path("docker/postgres/configs/000-pgconf-default")
    data["pool"]["path"] = Path("docker/worker_pool/configs/002-poolconf-4x8")
    data["rl"]["environment"] = {"address": "tcp://127.0.0.1:5000"}
    config = RlExperimentConfig.model_validate(data)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "training-tasks.json").write_text(
        '{"benchmark_id":"job","task_ids":["job-01a"]}'
    )
    return config, inputs, assets


@pytest.mark.parametrize(
    "address",
    [
        "http://host:5000",
        "tcp://host",
        "tcp://0.0.0.0:5000",
        "tcp://host:0",
        "tcp://host:5000/path",
        "tcp://user@host:5000",
    ],
)
def test_remote_address_rejection(address: str) -> None:
    with pytest.raises(ValueError):
        RemoteEnvironmentSettings(address=address)


def test_remote_translation_and_portable_identity(
    hybrid: tuple[RlExperimentConfig, Path, Path], tmp_path: Path
) -> None:
    config, inputs, assets = hybrid
    native = train.native_config(config, inputs)
    native = RLConfig.model_validate_json(native.model_dump_json())
    source = native.orchestrator.train.source[0]
    assert isinstance(source.env, QorlEnvironmentConfig)
    assert isinstance(source.env.taskset, QorlTasksetConfig)
    assert source.serve.address == "tcp://127.0.0.1:5000"
    assert source.env.postgres_config is None and source.env.pool_config is None
    assert source.env.taskset.selection == inputs / "training-tasks.json"
    expected = server.contract(config, inputs, assets)
    copied = tmp_path / "other-assets"
    copied.mkdir()
    for name in expected.renderer_assets:
        (copied / name).write_bytes((assets / name).read_bytes())
    assert server.contract(config, inputs, copied) == expected
    assert "model.safetensors" not in expected.renderer_assets
    server.validate_renderer(copied, expected.renderer)
    (copied / "tokenizer_config.json").write_text("{}")
    assert server.contract(config, inputs, copied) != expected
    assert native.orchestrator.model.client.wait_for_ready_timeout == 17


@pytest.mark.parametrize(
    "bind,advertised",
    [
        ("0.0.0.0", "http://127.0.0.1:8000/v1"),
        ("100.67.134.116", "http://127.0.0.1:8000/v1"),
    ],
)
def test_remote_endpoint_mismatch(
    hybrid: tuple[RlExperimentConfig, Path, Path], bind: str, advertised: str
) -> None:
    config, inputs, _ = hybrid
    data = config.model_dump()
    data["model"]["base_url"] = advertised
    data["inference"]["serving"]["host"] = bind
    with pytest.raises(ValueError, match=r"host|endpoint"):
        train.native_config(RlExperimentConfig.model_validate(data), inputs)


@pytest.mark.parametrize("thinking", [False, True])
def test_native_remote_episode_and_disconnect(
    hybrid: tuple[RlExperimentConfig, Path, Path],
    repository_root: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
) -> None:
    asyncio.run(
        remote_episode(
            hybrid, repository_root, postgres_config, pool_config, monkeypatch, thinking
        )
    )


async def remote_episode(
    hybrid: tuple[RlExperimentConfig, Path, Path],
    repository_root: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
) -> None:
    config, inputs, assets = hybrid
    bound = socket.socket()
    bound.bind(("127.0.0.1", 0))
    port = bound.getsockname()[1]
    body = config.model_dump()
    body["model"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    body["inference"]["serving"]["port"] = port
    body["inference"]["thinking"] = thinking
    config = RlExperimentConfig.model_validate(body)
    expected = server.contract(config, inputs, assets)
    native = RLConfig.model_validate_json(
        train.native_config(config, inputs).model_dump_json()
    )
    assert isinstance(expected.renderer, Qwen35RendererConfig)
    assert expected.renderer.enable_thinking is thinking
    assert isinstance(native.orchestrator.renderer, Qwen35RendererConfig)
    assert native.orchestrator.renderer.enable_thinking is thinking
    inference = InferenceClient(
        native.orchestrator.model.client,
        model_name=native.orchestrator.model.name,
        train_client_type="renderer",
        renderer_config=native.orchestrator.renderer,
    )
    tokenizer = server.load_tokenizer(str(assets))
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    generated: list[JsonObject] = []
    completions: list[list[int]] = []
    blocked = asyncio.Event()
    release = asyncio.Event()
    sql: list[str] = []
    active = QorlRuntime(
        TaskSet.load(repository_root, "job"),
        pool_config,
        "fake-remote",
        postgres_config,
    )

    def execute(command: list[str], query: str) -> subprocess.CompletedProcess[str]:
        sql.append(query)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "Plan": {"Node Type": "Result"},
                        "Execution Time": 10,
                        "Planning Time": 1,
                    }
                ]
            ),
            "",
        )

    for worker in active.workers:
        worker.client = PostgresClient(execute, active.settings, active.indexes)
    started: list[QorlEnvironment] = []
    stopped: list[QorlEnvironment] = []
    commits: list[Path] = []
    captures: list[QorlRuntime] = []
    identity_writes: list[Path] = []
    original_manifest = QorlRuntime.pool_manifest
    original_write = server.write_json
    original_native_write = harness_module.write_native
    loop_thread = threading.get_ident()
    writing_threads: list[int] = []

    def commit(path: Path) -> str:
        commits.append(path)
        return "test-commit"

    def manifest(active: QorlRuntime) -> PoolManifest:
        assert started and not stopped
        captures.append(active)
        return original_manifest(active)

    def write_identity(path: Path, value: JsonObject) -> None:
        if path.name == "identity.json":
            identity_writes.append(path)
        original_write(path, value)

    def write_trace(path: Path, trace: vf.Trace[vf.TaskData]) -> None:
        writing_threads.append(threading.get_ident())
        assert writing_threads[-1] != loop_thread
        original_native_write(path, trace)

    monkeypatch.setattr(server, "code_commit", commit)
    monkeypatch.setattr(QorlRuntime, "pool_manifest", manifest)
    monkeypatch.setattr(server, "write_json", write_identity)
    monkeypatch.setattr(harness_module, "write_native", write_trace)

    async def start(environment: QorlEnvironment) -> None:
        started.append(environment)

    async def stop(environment: QorlEnvironment) -> None:
        await shared_runtime.drain()
        stopped.append(environment)

    monkeypatch.setattr(QorlEnvironment, "start", start)
    monkeypatch.setattr(QorlEnvironment, "stop", stop)
    monkeypatch.setattr(shared_runtime, "current", lambda: active)
    monkeypatch.setattr(shared_runtime, "_runtime", active)
    monkeypatch.setattr(server, "CONTROL_TIMEOUT", 0.05)
    monkeypatch.setattr(server, "HEALTH_INTERVAL", 0.01)
    output = inputs / "service"
    output.mkdir()
    env = train.environment_config(config, inputs, config.model)
    harness = env.agent.harness
    assert isinstance(harness, QorlHarnessConfig)
    harness.evidence_directory = output
    service = server.QorlEnvServer(
        env,
        experiment=config,
        expected=expected,
        assets=assets,
        output=output,
        address="tcp://127.0.0.1:0",
    )
    assert len(commits) == 1 and not captures
    serving = asyncio.create_task(service.run())
    client = server.QorlEnvClient(service.address)
    app = web.Application()

    async def health(_: web.Request) -> web.Response:
        return web.json_response({})

    async def models_response(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "data": [
                    {
                        "id": inference.model_name,
                        "max_model_len": config.model.context_length,
                    }
                ]
            }
        )

    async def generate(request: web.Request) -> web.Response:
        from qorl.model.client import JSON_OBJECT

        generated.append(JSON_OBJECT.validate_json(await request.read()))
        if len(generated) > 2:
            blocked.set()
            await release.wait()
        text = (
            '<tool_call>\n<function=evaluate_candidate>\n<parameter=action>\n{"version":1}\n</parameter>\n</function>\n</tool_call><|im_end|>'
            if len(generated) == 1
            else "<tool_call>\n<function=finish>\n<parameter=selected_candidate_id>\ndefault\n</parameter>\n</function>\n</tool_call><|im_end|>"
        )
        tokens = encode(tokenizer, text)
        completions.append(tokens)
        return web.json_response(
            {
                "choices": [
                    {
                        "token_ids": tokens,
                        "sampling_mask": [
                            [token, (token + 1) % len(tokenizer)] for token in tokens
                        ],
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{token}", "logprob": -1.0}
                                for token in tokens
                            ]
                        },
                    }
                ]
            }
        )

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models_response)
    app.router.add_post("/inference/v1/generate", generate)
    http = web.AppRunner(app)
    await http.setup()
    try:
        await client.wait_for_server_startup(timeout=2)
        identity = await client.control(EnvironmentControlRequest(operation="identity"))
        assert identity.identity is not None and not identity.inference_ready
        for _ in range(4):
            assert await client.health()
            assert (
                await client.control(EnvironmentControlRequest(operation="identity"))
            ).identity == identity.identity
        assert len(commits) == len(captures) == 1 and not identity_writes
        assert len(started) == 1 and len(active.workers) == 4
        claim = EnvironmentClaim(
            run_id="one",
            contract=expected,
            renderer_source=inference.model_name,
            hostname="GPU-host",
            repository=Path("/gpu/repo"),
            training_directory=Path("/gpu/run/training"),
            code_commit="test",
        )
        with pytest.raises(RuntimeError, match="mismatch"):
            await client.control(
                EnvironmentControlRequest(
                    operation="claim",
                    claim=claim.model_copy(
                        update={
                            "contract": expected.model_copy(
                                update={"selection_sha256": "wrong"}
                            )
                        }
                    ),
                )
            )
        await client.control(EnvironmentControlRequest(operation="claim", claim=claim))
        with pytest.raises(RuntimeError, match="another training"):
            await client.control(
                EnvironmentControlRequest(
                    operation="claim", claim=claim.model_copy(update={"run_id": "two"})
                )
            )
        assert not generated and not sql  # Service was ready before inference existed.
        await web.SockSite(http, bound).start()
        response = await client.control(
            EnvironmentControlRequest(operation="probe", run_id="one")
        )
        assert response.inference_ready
        assert len(identity_writes) == 1
        episode = await asyncio.wait_for(dispatch(client, inference), timeout=20)
        assert episode.ok, episode.errors
        assert len(episode.traces) == 1
        trace = episode.traces[0]
        record = RlRolloutRecord.model_validate_json(json.dumps(trace.info["qorl"]))
        policy = AgentTrace.model_validate(trace.info["qorl_policy"])
        assert record.final is not None and record.final.kind == "kept_default"
        assert (
            len(record.candidates) == 1 and record.final.selected_candidate_id is None
        )
        assert sum("ANALYZE" in query for query in sql) == 2
        assert len(generated) == 2
        for request, response in zip(generated, policy.model_responses, strict=True):
            ids = request["token_ids"]
            assert isinstance(ids, list) and len(ids) == response.prompt_tokens
        assert [tool.function.name for tool in policy.model_requests[-1].tools] == [
            "finish"
        ]
        assert trace.num_output_tokens > 0
        sampled = [node for node in trace.nodes if node.sampled]
        assert len(sampled) == len(completions) == 2
        for node, tokens in zip(sampled, completions, strict=True):
            assert [
                token
                for token, mask in zip(node.token_ids, node.mask, strict=True)
                if mask
            ] == tokens
            assert node.logprobs == [-1.0] * len(tokens)
            assert node.sampling_mask is not None
            np.testing.assert_array_equal(node.sampling_mask.counts, [2] * len(tokens))
            np.testing.assert_array_equal(
                node.sampling_mask.ids,
                [
                    item
                    for token in tokens
                    for item in (token, (token + 1) % len(tokenizer))
                ],
            )
        assert all(not any(node.mask) for node in trace.nodes if not node.sampled)
        saved = vf.WireEpisode.model_validate(
            server.CODEC.unpackb(
                (output / "episodes" / f"{episode.id}.msgpack").read_bytes(), raw=False
            )
        )
        assert saved.traces[0].info["qorl"] == trace.info["qorl"]
        saved_sampled = [node for node in saved.traces[0].nodes if node.sampled]
        for before, after in zip(sampled, saved_sampled, strict=True):
            assert before.token_ids == after.token_ids and before.mask == after.mask
            assert before.logprobs == after.logprobs
            assert before.sampling_mask is not None and after.sampling_mask is not None
            np.testing.assert_array_equal(
                before.sampling_mask.ids, after.sampling_mask.ids
            )
            np.testing.assert_array_equal(
                before.sampling_mask.counts, after.sampling_mask.counts
            )
        assert list((output / "rollouts").glob("*.msgpack"))
        # Cancel a real native request while its harness is waiting on inference.
        pending = asyncio.create_task(dispatch(client, inference))
        await asyncio.wait_for(blocked.wait(), timeout=5)
        assert active.work
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        async with asyncio.timeout(2):
            while not all(cancel.is_set() for cancel in active.work.values()):
                await asyncio.sleep(0.01)
        assert await client.health() and not serving.done()
        release.set()
        async with asyncio.timeout(5):
            while active.work or service.episodes:
                await asyncio.sleep(0.01)
        assert await client.health() and len(generated) == 3
        retained = [
            vf.Trace[vf.TaskData].model_validate(
                server.CODEC.unpackb(path.read_bytes(), raw=False)
            )
            for path in (output / "rollouts").glob("*.msgpack")
        ]
        assert len(retained) == 2
        assert any(item.info["qorl"]["failure"] is not None for item in retained)
        before = len(captures)
        for _ in range(4):
            await client.control(EnvironmentControlRequest(operation="identity"))
        assert len(captures) == before and len(commits) == 1
        assert len(identity_writes) == 1 and len(writing_threads) == 2
        original_drain = shared_runtime.drain

        async def slow_drain() -> None:
            await asyncio.sleep(0.08)
            await original_drain()

        short = harness.model_copy(deep=True)
        assert short.model is not None
        short.measurement = short.measurement.model_copy(
            update={
                "default_timeout_seconds": 0.02,
                "candidate_timeout_floor_seconds": 0.02,
            }
        )
        short.model = short.model.model_copy(update={"request_timeout_seconds": 0.01})
        monkeypatch.setattr(train, "CANCELLATION_GRACE_SECONDS", 0.1)
        monkeypatch.setattr(shared_runtime, "drain", slow_drain)
        drain_timeout = train.cancellation_timeout(short, 1)
        assert drain_timeout > 0.08 > server.CONTROL_TIMEOUT
        cancelled = await client.control(
            EnvironmentControlRequest(operation="cancel", run_id="one"),
            timeout=drain_timeout,
        )
        assert cancelled.success and len(identity_writes) == 2
        assert cancelled.identity == identity.identity
        monitor = asyncio.create_task(client.monitor(identity.identity.service_id))
        serving.cancel()
        await serving
        with pytest.raises(RuntimeError, match="health timed out"):
            await asyncio.wait_for(monitor, timeout=2)
        assert stopped == started and not active.work
        with active.claim_worker() as released:
            assert released in active.workers
    finally:
        release.set()
        serving.cancel()
        with suppress(asyncio.CancelledError):
            await serving
        await client.close()
        await inference.aclose()
        await http.cleanup()
        bound.close()


def test_hanging_probe_is_bounded(
    hybrid: tuple[RlExperimentConfig, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, inputs, assets = hybrid
    expected = server.contract(config, inputs, assets)
    service = server.QorlEnvServer(
        train.environment_config(config, inputs, config.model),
        experiment=config,
        expected=expected,
        assets=assets,
        output=inputs,
        address="tcp://127.0.0.1:0",
    )
    service.claim = EnvironmentClaim(
        run_id="test",
        contract=expected,
        renderer_source="gpu-base",
        hostname="test",
        repository=inputs,
        training_directory=inputs,
        code_commit="test",
    )
    assert service.startup_timeout == 17
    service.startup_timeout = 1

    async def hanging(*_: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "check_inference_ready", hanging)
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(service.probe())
        assert not service.inference_ready
    finally:
        service.frontend.close()
        service.ctx.term()


@pytest.mark.parametrize(
    ("floor", "http_seconds", "attempts", "rollouts", "capacity", "expected"),
    [
        (5.0, 2.0, 1, 4, 8, 910.0),
        (1200.0, 2.0, 1, 4, 8, 1210.0),
        (5.0, 300.0, 3, 4, 8, 1930.0),
        (5.0, 300.0, 3, 4, 4, 1930.0),
        (5.0, 300.0, 3, 4, 2, 7690.0),
        (5.0, 300.0, 3, 1, 1, 1930.0),
    ],
)
def test_cancellation_deadlines_include_candidate_floor_multiplier_and_http_retries(
    hybrid: tuple[RlExperimentConfig, Path, Path],
    floor: float,
    http_seconds: float,
    attempts: int,
    rollouts: int,
    capacity: int,
    expected: float,
) -> None:
    config, inputs, _ = hybrid
    harness = train.environment_config(config, inputs, config.model).agent.harness
    assert isinstance(harness, QorlHarnessConfig)
    body = harness.model_dump()
    body["measurement"].update(
        default_timeout_seconds=300.0,
        candidate_timeout_floor_seconds=floor,
        candidate_timeout_multiplier=3.0,
    )
    body["model"]["request_timeout_seconds"] = http_seconds
    body["model"]["max_concurrent_requests"] = capacity
    body["model"]["retry"].update(max_attempts=attempts, maximum_delay_seconds=30.0)
    resolved = QorlHarnessConfig.model_validate(body)
    assert train.cancellation_timeout(resolved, rollouts) == expected


def test_service_startup_failure_closes_owned_resources(
    hybrid: tuple[RlExperimentConfig, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, inputs, assets = hybrid
    services: list[server.QorlEnvServer] = []
    original_run = server.QorlEnvServer.run
    handlers = list(logging.getLogger().handlers)
    level = logging.getLogger().level

    async def failing_start(_: QorlEnvironment) -> None:
        raise RuntimeError("PostgreSQL startup failed")

    async def run(service: server.QorlEnvServer) -> None:
        services.append(service)
        await original_run(service)

    def load(_: Path) -> RlExperimentConfig:
        return config

    monkeypatch.setattr(server, "load_config", load)
    monkeypatch.setattr(QorlEnvironment, "start", failing_start)
    monkeypatch.setattr(server.QorlEnvServer, "run", run)
    assert config.rl.environment is not None
    with pytest.raises(RuntimeError, match="PostgreSQL startup failed"):
        asyncio.run(
            server.serve_environment(
                inputs, config.rl.environment.address, assets, inputs / "failed-service"
            )
        )
    assert len(services) == 1
    assert services[0].frontend.closed and services[0].ctx.closed
    assert logging.getLogger().handlers == handlers
    assert logging.getLogger().level == level


def test_service_cli_forwards_explicit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str, Path, Path]] = []

    def serve(experiment: Path, address: str, assets: Path, output: Path) -> None:
        calls.append((experiment, address, assets, output))

    monkeypatch.setattr(server, "serve", serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qorl",
            "rl",
            "serve",
            "experiments/new",
            "--bind",
            "tcp://100.84.223.59:5000",
            "--renderer-model",
            "/pg/assets",
            "--output",
            "/pg/service",
        ],
    )
    assert cli.main() == 0
    assert calls == [
        (
            Path("experiments/new"),
            "tcp://100.84.223.59:5000",
            Path("/pg/assets"),
            Path("/pg/service"),
        )
    ]


@pytest.mark.parametrize("conflict", ["lock", "pool"])
def test_service_rejects_competing_pool_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: str
) -> None:
    def lock(fd: object, operation: int) -> None:
        if conflict == "lock":
            raise BlockingIOError("already locked")

    def docker(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["docker", "ps", "--format", "{{.Image}}"]
        assert check and capture_output and text
        return subprocess.CompletedProcess(command, 0, "qorl-postgres:test\n", "")

    def forbidden(*_: object) -> None:
        pytest.fail("conflicting service reached environment startup")

    monkeypatch.setattr(server.fcntl, "flock", lock)
    monkeypatch.setattr(server.subprocess, "run", docker)
    monkeypatch.setattr(server, "serve_environment", forbidden)
    with pytest.raises(RuntimeError, match=r"another|already"):
        server.serve(tmp_path, "tcp://127.0.0.1:5000", tmp_path, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "failure",
    ["startup", "identity", "probe", "disconnect", "drain_disconnect", "write", "none"],
)
def test_remote_launcher_lifecycle(
    hybrid: tuple[RlExperimentConfig, Path, Path],
    repository_root: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config, inputs, assets = hybrid
    native = train.native_config(config, inputs)
    expected = server.contract(config, inputs, assets)
    claim = EnvironmentClaim(
        run_id="launcher",
        contract=expected,
        renderer_source=str(assets),
        hostname="GPU",
        repository=inputs,
        training_directory=native.run_dir,
        code_commit="test",
    )
    active = QorlRuntime(
        TaskSet.load(repository_root, "job"), pool_config, "fake", postgres_config
    )
    identity = EnvironmentServiceIdentity(
        service_id="service",
        hostname="PG",
        repository=repository_root,
        renderer_model=assets,
        evidence_directory=inputs,
        address="tcp://127.0.0.1:5000",
        contract=expected,
        code_commit="test",
        database_pool=active.pool_manifest(),
    )
    operations: list[str] = []
    closed: list[bool] = []
    processes: list[asyncio.subprocess.Process] = []
    spawned = asyncio.create_subprocess_exec
    write = train.write_json
    ready = asyncio.Event()
    cancellation_started = asyncio.Event()

    class Client:
        def __init__(self, address: str) -> None:
            assert address == identity.address

        async def wait_for_server_startup(self, timeout: float) -> None:
            assert timeout == 17
            if failure == "startup":
                raise TimeoutError("startup failed")

        async def control(
            self,
            request: EnvironmentControlRequest,
            *,
            timeout: float = server.CONTROL_TIMEOUT,
        ) -> EnvironmentControlResponse:
            operations.append(request.operation)
            if request.operation == "claim":
                if failure == "identity":
                    return EnvironmentControlResponse()
                return EnvironmentControlResponse(identity=identity)
            assert request.run_id == claim.run_id
            if request.operation == "probe":
                assert timeout == 17 + server.CONTROL_TIMEOUT
                if failure == "probe":
                    raise TimeoutError("reverse readiness failed")
                ready.set()
                return EnvironmentControlResponse(
                    identity=identity, inference_ready=True
                )
            assert request.operation == "cancel"
            environment = native.orchestrator.train.source[0].env
            assert isinstance(environment, QorlEnvironmentConfig)
            harness = environment.agent.harness
            assert isinstance(harness, QorlHarnessConfig)
            expected_timeout = (
                server.CONTROL_TIMEOUT
                if failure in {"startup", "identity", "disconnect", "write"}
                else train.cancellation_timeout(harness, config.training.max_inflight)
            )
            assert timeout == expected_timeout
            cancellation_started.set()
            if failure == "drain_disconnect":
                await asyncio.Event().wait()
            if failure in {"disconnect", "startup"}:
                raise TimeoutError("remote unavailable during cleanup")
            return EnvironmentControlResponse(identity=identity)

        async def monitor(self, service_id: str) -> None:
            assert service_id == identity.service_id
            await ready.wait()
            if failure in {"disconnect", "write"}:
                raise RuntimeError("remote disconnected")
            if failure == "drain_disconnect":
                await cancellation_started.wait()
                raise RuntimeError("remote disconnected during drain")
            await asyncio.Event().wait()

        async def close(self) -> None:
            closed.append(True)

    async def spawn(
        *command: str, cwd: Path, env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        assert command[1:4] == ("-m", "prime_rl.entrypoints.rl", "@")
        assert env["VIRTUAL_ENV"] == sys.prefix
        assert env["PATH"].split(":")[0] == str(Path(sys.executable).parent)
        assert env["CUDA_VISIBLE_DEVICES"] == "3,7"
        process = await spawned(
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)"
            if failure in {"none", "drain_disconnect"}
            else "import time; time.sleep(60)",
            cwd=cwd,
            env=env,
        )
        processes.append(process)
        return process

    def forbidden(*_: object, **__: object) -> None:
        pytest.fail("remote launcher constructed a local environment")

    def write_evidence(path: Path, value: JsonObject) -> None:
        if path.name == "remote-cleanup.json" and failure == "write":
            assert closed == [True]
            assert processes and all(
                process.returncode is not None for process in processes
            )
            raise OSError("output disk full")
        write(path, value)

    monkeypatch.setattr(server, "QorlEnvClient", Client)
    monkeypatch.setattr(train, "EnvServer", forbidden)
    monkeypatch.setattr(train, "write_json", write_evidence)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if failure in {"none", "drain_disconnect"}:
        asyncio.run(train.launch(native, [3, 7], claim))
    else:
        with pytest.raises((TimeoutError, RuntimeError, ValueError, OSError)):
            asyncio.run(train.launch(native, [3, 7], claim))
    assert closed == [True] and operations[-1] == "cancel"
    assert len(processes) == (0 if failure in {"startup", "identity"} else 1)
    assert all(process.returncode is not None for process in processes)
    if failure == "write":
        assert not (native.run_dir / "remote-cleanup.json").exists()
    else:
        cleanup = EnvironmentCleanup.model_validate_json(
            (native.run_dir / "remote-cleanup.json").read_bytes()
        )
        assert cleanup.cancellation_acknowledged == (
            failure not in {"startup", "disconnect", "drain_disconnect"}
        )
    if processes:
        recorded = RLConfig.model_validate_json(
            (native.run_dir / train.NATIVE_CONFIG).read_bytes()
        )
        taskset = recorded.orchestrator.train.source[0].env.taskset
        assert (
            isinstance(taskset, QorlTasksetConfig)
            and taskset.remote_run == claim.run_id
        )
