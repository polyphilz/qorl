import asyncio
import json
import subprocess
import tomllib
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
import verifiers.v1 as vf

from qorl.agent.schemas import AgentTrace
from qorl.agent.types import StopReason
from qorl.model.client import HttpTransport
from qorl.model.exceptions import ModelError
from qorl.model.schemas import JsonObject
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.rl import runtime as shared_runtime
from qorl.rl.harness import QorlHarness
from qorl.rl.runtime import QorlRuntime
from qorl.rl.schemas import QorlHarnessConfig, QorlTaskData, RlRolloutRecord
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig

WAIT_SECONDS = 2.0


@pytest.mark.parametrize("model_fails", [False, True])
def test_actual_harness_uses_proxy_and_retains_records(
    repository_root: Path,
    postgres_config: PostgresConfig,
    pool_config: PoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    model_fails: bool,
) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    config = QorlHarnessConfig.model_validate(
        {
            key: defaults[key]
            for key in ("agent", "measurement", "rl", "model", "inference")
        }
    )
    assert config.model is not None
    runtime = QorlRuntime(
        TaskSet.load(repository_root, "job"), pool_config, "test", postgres_config
    )
    task = runtime.task_set.tasks[0]
    worker = runtime.workers[0]

    def execute(command: list[str], query: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "Plan": {"Node Type": "Result"},
                        "Execution Time": 1,
                        "Planning Time": 1,
                    }
                ]
            ),
            "",
        )

    worker.client = PostgresClient(execute, runtime.settings, runtime.indexes)
    monkeypatch.setattr(shared_runtime, "current", Mock(return_value=runtime))
    requests: list[tuple[str, str, JsonObject]] = []

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        assert body is not None and config.model is not None
        requests.append((transport.base_url, path, body))
        if path == "../tokenize":
            return {"count": 1, "max_model_len": config.model.context_length}
        assert path == "chat/completions"
        if model_fails:
            raise ModelError("provider unavailable")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "keep",
                                "function": {"name": "keep_default", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(HttpTransport, "request", request)
    trace = Mock(spec=vf.Trace, id="trace", info={})
    arguments = (
        Mock(spec=vf.ModelContext, model="training-model"),
        trace,
        Mock(),
        "http://proxy.test/v1",
        "rollout-secret",
        {},
        QorlTaskData(task_id=task.task_id, template_id=task.template_id),
    )
    harness = QorlHarness(config)
    if model_fails:
        with pytest.raises(ModelError, match="provider unavailable"):
            asyncio.run(harness.launch(*arguments))
    else:
        assert asyncio.run(harness.launch(*arguments)).exit_code == 0
    assert [(url, path) for url, path, _ in requests] == [
        ("http://proxy.test/v1", "../tokenize"),
        ("http://proxy.test/v1", "chat/completions"),
    ]
    assert requests[0][2]["messages"] == requests[1][2]["messages"]
    assert "seed" not in requests[1][2]
    assert worker.client.explain_analyze_calls == 2
    record = RlRolloutRecord.model_validate(trace.info["qorl"])
    policy_trace = AgentTrace.model_validate(trace.info["qorl_policy"])
    assert record.task_id == task.task_id
    assert record.database_worker == worker.resources.manifest()
    if model_fails:
        assert record.final is None and record.failure is not None
        assert policy_trace.stop_reason is None
    else:
        assert record.final is not None and record.final.speedup == 1
        assert policy_trace.stop_reason == StopReason.MODEL_KEEP_DEFAULT


def test_fallback_config_matches_rl_defaults(repository_root: Path) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    expected = QorlHarnessConfig.model_validate(
        {key: defaults[key] for key in ("agent", "measurement", "rl")}
    )

    fallback = QorlHarnessConfig(id="qorl")

    assert fallback.agent == expected.agent
    assert fallback.measurement == expected.measurement
    assert fallback.rl == expected.rl


@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_cancellation_waits_for_worker_cleanup(
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancellations: int,
    worker_fails: bool,
) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    harness = QorlHarness(
        QorlHarnessConfig.model_validate(
            {key: defaults[key] for key in ("agent", "measurement", "rl")}
        )
    )
    started, cancellation_seen, release_worker, cleaned_up = (
        Event(),
        Event(),
        Event(),
        Event(),
    )

    def run(
        ctx: Mock, trace: Mock, endpoint: str, secret: str, data: Mock, cancel: Event
    ) -> None:
        started.set()
        assert cancel.wait(WAIT_SECONDS)
        cancellation_seen.set()
        assert release_worker.wait(WAIT_SECONDS)
        cleaned_up.set()
        if worker_fails:
            raise CancelledError("rollout cancelled")

    monkeypatch.setattr(harness, "_run", run)
    monkeypatch.setattr(shared_runtime, "current", Mock(return_value=Mock(work={})))

    async def check() -> None:
        work = asyncio.create_task(
            harness.launch(
                Mock(),
                Mock(info={}),
                Mock(),
                "unused",
                "unused",
                {},
                QorlTaskData(task_id="job-01a", template_id="job-01"),
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, WAIT_SECONDS)
            for _ in range(cancellations):
                work.cancel()
                assert await asyncio.to_thread(cancellation_seen.wait, WAIT_SECONDS)
                # Let the cancellation handler run while the worker still owns its slot.
                await asyncio.sleep(0)
                assert not work.done()
                assert not cleaned_up.is_set()
        finally:
            release_worker.set()
            with pytest.raises(asyncio.CancelledError):
                await work
        assert cleaned_up.is_set()

    asyncio.run(check())
