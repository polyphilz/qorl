"""Script external boundaries while retaining the real pool, client, agent and evaluator."""

import json
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier

import pytest

from qorl.evaluation import evaluate
from qorl.experiment.create import latest_template
from qorl.experiment.schemas import (
    EvaluationExperimentConfig,
    ExperimentMethod,
    load_config,
)
from qorl.model.client import HttpTransport, LocalModelClient
from qorl.model.exceptions import ModelRequestError
from qorl.model.schemas import (
    AdvertisedModel,
    JsonObject,
    LocalInferenceSettings,
    LocalServerIdentity,
    ModelSettings,
)
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.postgres.schemas import PostgresIndexes
from qorl.worker_pool.containers import ContainerPool
from qorl.worker_pool.schemas import PoolConfig, WorkerSlot

WAIT_SECONDS = 5.0


@dataclass
class EvaluationActivity:
    mode: str = "measured"
    pools: list[ContainerPool] = field(default_factory=list[ContainerPool])
    closed: list[ContainerPool] = field(default_factory=list[ContainerPool])
    served: list[ModelSettings] = field(default_factory=list[ModelSettings])
    server_closed: bool = False
    queries: list[str] = field(default_factory=list[str])
    requests: list[JsonObject] = field(default_factory=list[JsonObject])
    captures: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    claimed_workers: set[int] = field(default_factory=set[int])
    barrier: Barrier | None = None

    def command(
        self, command: list[str], query: str
    ) -> subprocess.CompletedProcess[str]:
        """Return one stable plan per role and preserve the actual generated SQL."""
        self.queries.append(query)
        if not query.startswith("EXPLAIN"):
            return subprocess.CompletedProcess(command, 0, "{}", "")
        candidate = "/*+" in query
        if self.mode == "default_timeout" and not candidate:
            return subprocess.CompletedProcess(
                command, 1, "", "canceling statement due to statement timeout"
            )
        if self.mode == "candidate_timeout" and candidate:
            return subprocess.CompletedProcess(
                command, 1, "", "canceling statement due to statement timeout"
            )
        novel = candidate and self.mode not in {"non_novel", "duplicate"}
        plan: JsonObject = {"Node Type": "Result", "Plan Rows": 1}
        if novel:
            plan = {"Node Type": "Materialize", "Plans": [plan]}
        document: JsonObject = {
            "Plan": plan,
            "Execution Time": 5.0 if candidate else 100.0,
            "Planning Time": 1.0,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps([document]),
            "HintStateDump: {used hints:Set(seq_page_cost)}, {not used hints:(none)}, {duplicate hints:(none)}, {error hints:(none)}"
            if candidate
            else "",
        )

    def request(
        self, transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        """Exercise local chat and Astra Responses through their actual translators."""
        assert body is not None
        self.requests.append(body)
        if path == "../tokenize":
            return {"count": 10, "max_model_len": 20_480}
        if path == "responses/input_tokens":
            return {"input_tokens": 10}
        if self.mode == "provider_failure":
            raise ModelRequestError("provider unavailable")
        if self.mode == "interrupt":
            raise KeyboardInterrupt()
        history = body.get("messages", body.get("input"))
        assert isinstance(history, list)
        results = [
            message
            for message in history
            if isinstance(message, dict)
            and (
                message.get("role") == "tool"
                or message.get("type") == "function_call_output"
            )
        ]
        if self.barrier is not None and not results:
            self.barrier.wait(timeout=WAIT_SECONDS)
        if self.mode == "keep":
            name, arguments = "keep_default", "{}"
        elif not results:
            name, arguments = "get_plan", '{"candidate_id":"default"}'
        elif len(results) == 1:
            action: JsonObject = (
                {"version": 2}
                if self.mode == "invalid"
                else {"version": 1}
                if self.mode == "duplicate"
                else {"version": 1, "settings": {"seq_page_cost": 2.0}}
            )
            name, arguments = "evaluate_candidate", json.dumps({"action": action})
        else:
            name, arguments = "finish", "{}"
        if self.mode == "provider_failure_after_candidate" and len(results) == 2:
            raise ModelRequestError("provider unavailable after validation")
        if path == "chat/completions":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"call-{len(results)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        },
                        "finish_reason": "length"
                        if self.mode == "truncated"
                        else "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        assert path == "responses"
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call-{len(results)}",
                    "name": name,
                    "arguments": arguments,
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


@pytest.fixture
def evaluation_config() -> EvaluationExperimentConfig:
    config = load_config(latest_template(ExperimentMethod.EVAL))
    assert isinstance(config, EvaluationExperimentConfig)
    return config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"name_or_path": "example/base", "context_length": 20_480}
            ),
            "inference": config.inference.model_copy(update={"max_tokens": 2048}),
        }
    )


@pytest.fixture
def activity(monkeypatch: pytest.MonkeyPatch) -> EvaluationActivity:
    observed = EvaluationActivity()

    def start(
        name: str,
        archive: Path,
        *,
        postgres_config: PostgresConfig,
        pool_config: PoolConfig,
    ) -> ContainerPool:
        pool = ContainerPool(name, pool_config, postgres_config)
        for worker in pool.workers:
            worker.client = PostgresClient(
                observed.command, pool.settings, PostgresIndexes(by_table={})
            )
        observed.pools.append(pool)
        return pool

    def close(pool: ContainerPool) -> None:
        assert not observed.claimed_workers
        observed.closed.append(pool)

    original_claim = ContainerPool.claim_worker

    @contextmanager
    def claim(pool: ContainerPool) -> Generator[WorkerSlot]:
        with original_claim(pool) as slot:
            observed.claimed_workers.add(slot.resources.index)
            try:
                yield slot
            finally:
                observed.claimed_workers.remove(slot.resources.index)

    def capture(
        pool: ContainerPool, slot: WorkerSlot, output: Path, phase: str
    ) -> None:
        observed.captures.append((slot.resources.index, phase))

    @contextmanager
    def serve(
        model: ModelSettings,
        inference: LocalInferenceSettings,
        gpu_ids: list[int],
        log_path: Path,
    ) -> Generator[LocalModelClient]:
        observed.served.append(model)
        client = LocalModelClient(model, inference)
        advertised = AdvertisedModel(
            id=model.name_or_path, max_model_len=model.context_length
        )
        client.identity = LocalServerIdentity(
            base_url=model.base_url or "",
            model=advertised,
            context_model=advertised,
            vllm_version="test-runtime",
        )
        try:
            yield client
        finally:
            observed.server_closed = True

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        return observed.request(transport, path, body)

    monkeypatch.setattr(evaluate, "start_pool", start)
    monkeypatch.setattr(ContainerPool, "close", close)
    monkeypatch.setattr(ContainerPool, "claim_worker", claim)
    monkeypatch.setattr(evaluate, "capture_environment", capture)
    monkeypatch.setattr(evaluate, "serve_local_model", serve)
    monkeypatch.setattr(HttpTransport, "request", request)
    return observed
