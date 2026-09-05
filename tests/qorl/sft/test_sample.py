from __future__ import annotations

from pathlib import Path

import pytest
from tests.qorl.sft.factories import baseline, sample

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.db.container import PostgresContainer
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.db.resources import DEFAULT_POOL_CONFIG, load_runtime_profile
from qorl.db.worker import PostgresWorker
from qorl.measure.schemas import Baseline, RunStatus
from qorl.sft.sample import (
    PlanValidationEvaluator,
    SampleRequest,
    evaluate_request,
    sample_limit,
    sampling_summary,
)
from qorl.sft.schemas import (
    DatasetConfig,
    SamplerIdentity,
    SamplingMode,
    load_record,
)
from qorl.workload.taskset import TaskSet


def config() -> DatasetConfig:
    return load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )


def test_sampling_summary_deduplicates_fingerprints_per_task() -> None:
    summary = sampling_summary([sample(), sample(2)])

    assert summary.novel_candidates == 2
    assert summary.distinct_novel_fingerprints == 1
    assert summary.distinct_novel_fingerprint_yield == 0.5


def test_normal_and_default_best_sampling_caps_are_enforced() -> None:
    assert sample_limit(config(), SamplingMode.NORMAL, 5, 2) == 6
    assert sample_limit(config(), SamplingMode.DEFAULT_BEST, 7, 24) == 30
    with pytest.raises(RuntimeError, match="above its 6-sample cap"):
        sample_limit(config(), SamplingMode.NORMAL, 5, 3)
    with pytest.raises(RuntimeError, match="begin after the normal sample cap"):
        sample_limit(config(), SamplingMode.DEFAULT_BEST, 6, 1)


def test_unexpected_rollout_error_is_recorded(
    repository_root: Path,
    database_fixture: DatabaseFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = baseline()

    def start(evaluator: PlanValidationEvaluator) -> Baseline:
        evaluator.default = default
        return default

    def fail(policy: QoAgentPolicy, evaluator: object) -> None:
        del policy, evaluator
        raise TypeError("unexpected sampling failure")

    monkeypatch.setattr(PlanValidationEvaluator, "start", start)
    monkeypatch.setattr(QoAgentPolicy, "search", fail)
    monkeypatch.setattr(PostgresWorker, "task_indexes", lambda *_: {})
    agent_config = QoAgentConfig(
        model="model",
        revision="revision",
        vllm_version="version",
        use_flashinfer_sampler=False,
        enable_prefix_caching=True,
        base_url="http://127.0.0.1:8000/v1",
        context_length=20_480,
        maximum_model_turns=64,
        request_timeout_seconds=300,
        seed=7,
        sampling={"max_tokens": 2_048},
        thinking=False,
        tool_call_parser="qwen3_coder",
    )
    task_set = TaskSet.load(repository_root, "ceb")
    task = task_set.inventory["tasks"][0]
    request = SampleRequest(
        task=task,
        task_id=task["task_id"],
        template_id=task["template_id"],
        sample=1,
        seed=1,
        sampling_mode=SamplingMode.NORMAL,
    )
    profile = load_runtime_profile(repository_root, DEFAULT_POOL_CONFIG)
    resources = profile.workers[0]
    container = PostgresContainer(
        database_fixture, "test-sft-sample", profile, resources
    )
    pool = WorkerPool(
        (WorkerSlot(resources, container, PostgresWorker(container)),),
        "test-pool",
        "test-sha",
    )

    _, record = evaluate_request(
        pool,
        task_set,
        request,
        agent_config,
        None,
        SamplerIdentity(model="model", manifest_sha256="sha", server_identity={}),
    )

    assert record.status == RunStatus.FAILED
    assert record.default == default
    assert record.error is not None
    assert record.error.type == "TypeError"
    assert record.error.message == "unexpected sampling failure"
