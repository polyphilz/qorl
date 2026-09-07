"""Transient retries do not consume agent turns or bypass concurrency limits."""

import asyncio
import io
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPMessage
from unittest.mock import AsyncMock, Mock

import pytest
import verifiers.v1 as vf
from verifiers.v1.clients import EvalClient, EvalClientConfig, ModelContext
from verifiers.v1.clients import train as proxy_train
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage, Response

from qorl.model import client
from qorl.model.client import HttpTransport, retry_after_seconds
from qorl.model.exceptions import ModelError, ModelRequestError, TransientModelError
from qorl.model.schemas import JsonObject, ModelProvider, ModelSettings, RetrySettings

REQUEST_ATTEMPTS = 3
CONCURRENCY = 2
WORKERS = 4
WAIT_SECONDS = 5


def settings() -> ModelSettings:
    return ModelSettings(
        provider=ModelProvider.OPENAI,
        name_or_path="gpt-6-astra",
        context_length=1,
        base_url="https://example.test/v1",
        request_timeout_seconds=1,
        max_concurrent_requests=CONCURRENCY,
        retry=RetrySettings(max_attempts=REQUEST_ATTEMPTS),
    )


def http_error(status: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = HTTPMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.test/v1/responses",
        status,
        "failure",
        headers,
        io.BytesIO(b"temporary failure"),
    )


@pytest.mark.parametrize("status", [429, 500, 503, 529])
def test_retries_transient_failures_with_backoff(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    opened = Mock(
        side_effect=[
            http_error(status),
            http_error(status),
            io.BytesIO(b'{"result": true}'),
        ]
    )
    slept = Mock()
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.setattr(client.time, "sleep", slept)

    def upper_bound(lower: float, upper: float) -> float:
        return upper

    monkeypatch.setattr(client.random, "uniform", upper_bound)
    body: JsonObject = {"input": "unchanged"}
    assert HttpTransport(settings()).request("responses", body) == {"result": True}
    assert opened.call_count == REQUEST_ATTEMPTS
    assert [call.args[0] for call in slept.call_args_list] == [2.0, 4.0]
    assert len({call.args[0].data for call in opened.call_args_list}) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_fatal_http_errors_are_never_retried(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    opened = Mock(side_effect=http_error(status))
    slept = Mock()
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.setattr(client.time, "sleep", slept)
    with pytest.raises(ModelRequestError):
        HttpTransport(settings()).request("responses", {})
    assert opened.call_count == 1
    slept.assert_not_called()


def test_retry_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = Mock(side_effect=[http_error(429) for _ in range(REQUEST_ATTEMPTS)])
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.setattr(client.time, "sleep", Mock())
    with pytest.raises(TransientModelError):
        HttpTransport(settings()).request("responses", {})
    assert opened.call_count == REQUEST_ATTEMPTS


def test_retry_after_is_a_minimum_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = Mock(side_effect=[http_error(429, "10"), io.BytesIO(b"{}")])
    slept = Mock()
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.setattr(client.time, "sleep", slept)
    monkeypatch.setattr(client.random, "uniform", Mock(return_value=0))
    HttpTransport(settings()).request("responses", {})
    slept.assert_called_once_with(10.0)


def test_long_retry_after_fails_instead_of_retrying_too_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = Mock(side_effect=http_error(503, "600"))
    slept = Mock()
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    monkeypatch.setattr(client.time, "sleep", slept)
    with pytest.raises(TransientModelError):
        HttpTransport(settings()).request("responses", {})
    assert opened.call_count == 1
    slept.assert_not_called()


def test_invalid_json_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = Mock(return_value=io.BytesIO(b"not-json"))
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)
    with pytest.raises(ModelError, match="invalid JSON"):
        HttpTransport(settings()).request("responses", {})
    assert opened.call_count == 1


@pytest.mark.parametrize("value", [None, "garbage", "NaN", "inf"])
def test_invalid_retry_after_is_ignored(value: str | None) -> None:
    assert retry_after_seconds(value) is None


def test_semaphore_bounds_actual_http_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    capacity = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0

    def open_request(
        request: client.urllib.request.Request, timeout: int
    ) -> io.BytesIO:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == CONCURRENCY:
                capacity.set()
        try:
            assert release.wait(WAIT_SECONDS)
            return io.BytesIO(b"{}")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(client.urllib.request, "urlopen", open_request)
    transport = HttpTransport(settings())
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(transport.request, "responses", {}) for _ in range(WORKERS)
        ]
        try:
            assert capacity.wait(WAIT_SECONDS)
        finally:
            release.set()
        assert [future.result() for future in futures] == [{}] * WORKERS
    assert peak == CONCURRENCY


def test_proxy_retry_records_only_the_completion_received_by_qorl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lose a completed HTTP reply; the real proxy must replay, not sample again."""
    replies = [
        Response(
            id=name,
            created=0,
            model="test-model",
            message=AssistantMessage(content=name),
            finish_reason="stop",
        )
        for name in ("first", "second")
    ]
    for reply in replies:
        reply.raw = client.JSON_OBJECT.validate_python(
            proxy_train.serialize_completion(reply, reply.model)
        )
    generate = AsyncMock(side_effect=replies)
    monkeypatch.setattr(EvalClient, "get_response", generate)
    monkeypatch.setattr(client.time, "sleep", Mock())
    urlopen = client.urllib.request.urlopen
    keys: list[str | None] = []

    def lose_first_response(
        request: client.urllib.request.Request, timeout: int
    ) -> io.BytesIO:
        key = request.get_header("Idempotency-key")
        keys.append(key)
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
        if len(keys) == 1:
            raise TimeoutError("reply lost after the proxy committed its completion")
        return io.BytesIO(data)

    monkeypatch.setattr(client.urllib.request, "urlopen", lose_first_response)

    async def check() -> None:
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type="test", data=vf.TaskData()),
        )
        session = RolloutSession(
            ctx=ModelContext(
                model="test-model",
                client=EvalClientConfig(
                    base_url="http://unused.test/v1", api_key_var="QORL_TEST_MODEL_KEY"
                ),
            ),
            trace=trace,
        )
        async with InterceptionServer() as server, server.acquire(session) as slot:
            base_url, secret, _ = slot
            transport = HttpTransport(
                settings().model_copy(update={"base_url": f"{base_url}/v1"}),
                api_key=secret,
            )
            body: JsonObject = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply."}],
            }
            first = await asyncio.to_thread(transport.request, "chat/completions", body)
            assert first["id"] == "first"
            assert generate.await_count == len(trace.calls) == len(trace.branches) == 1
            assert keys[0]
            assert keys[0] == keys[1]
            # An independent call with the identical body is a new sample, not a retry.
            second = await asyncio.to_thread(
                transport.request, "chat/completions", body
            )
            assert second["id"] == "second"
            assert (
                generate.await_count
                == len(trace.calls)
                == len(trace.branches)
                == len(replies)
            )
            assert keys[-1] != keys[0]

    asyncio.run(check())
