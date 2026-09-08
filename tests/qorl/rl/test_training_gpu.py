"""Opt-in real policy rollouts, optimizer updates, weight handoff and evaluation."""

import os
from pathlib import Path

import pytest
import tomli_w
import torch
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from pydantic import BaseModel
from safetensors.torch import load

from qorl.adapters.export import checkpoint_sha256
from qorl.agent.schemas import AgentTrace
from qorl.evaluation.schemas import EvaluationReport
from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CreateRequest,
    ExperimentMethod,
    RlExperimentConfig,
    RunRequest,
    RunStage,
    load_config,
)
from qorl.model.schemas import JsonValue
from qorl.rl.report import RlTrainingReport
from qorl.rl.train import checkpoint_model
from qorl.taskset.schemas import TaskRole
from qorl.util.hashing import sha256_file

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_RL_GPU") != "1",
    reason="set QORL_TEST_RL_GPU=1 in the agreed isolated benchmark-host checkout",
)


class RecordedTrace(BaseModel):
    info: dict[str, JsonValue]


class RecordedEpisode(BaseModel):
    traces: list[RecordedTrace]


def test_real_rl_updates_and_explicit_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Path(os.environ["QORL_TEST_MODEL_PATH"]).resolve()
    # Keep the authored experiment and every attempt beside its evidence.
    root = Path(os.environ.get("QORL_TEST_RL_OUTPUT", str(tmp_path))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", root / "experiments")
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", root / "outputs")
    directory = create.create_experiment(
        CreateRequest(
            name="slice10-smoke",
            method=ExperimentMethod.RL,
            base_model_name_or_path=str(base),
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
            tasksets=("train=job[01:1]", "validation=job[02:1]"),
        )
    )
    loaded = load_config(directory / "config.toml")
    assert isinstance(loaded, RlExperimentConfig)
    document = loaded.model_dump(mode="json", exclude_none=True)
    document["training"].update(
        batch_size=4,
        group_size=4,
        max_steps=4,
        max_off_policy_steps=0,
        max_inflight=4,
        renderer={"name": "qwen3.5"},
        checkpoints={"type": "interval", "interval": 1},
    )
    document["agent"]["maximum_model_turns"] = 4
    document["evaluation"]["rollouts_per_task"] = 1
    config = RlExperimentConfig.model_validate(document)
    (directory / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    output = run.run_experiment(directory, RunRequest(RunStage.TRAIN))
    training = output / "training"
    report = RlTrainingReport.model_validate_json(
        (training / "report.json").read_bytes()
    )
    assert report.completed and report.optimizer_steps == [1, 2, 3, 4]
    checked_turns = 0
    stream_path = get_trace_stream(training)
    for chunk in chunk_numbers(stream_path):
        with open_chunk(stream_path, chunk) as stream:
            for line in stream:
                episode = RecordedEpisode.model_validate_json(line)
                for trace in episode.traces:
                    if trace.info.get("qorl_policy") is None:
                        continue
                    policy = AgentTrace.model_validate(trace.info["qorl_policy"])
                    for response in policy.model_responses:
                        assert response.prompt_tokens == response.usage.prompt_tokens
                        assert response.request["tool_choice"] == "auto"
                        assert response.request["parallel_tool_calls"] is True
                        checked_turns += 1
    assert checked_turns > 0
    assert any(
        item.assigned_advantage is not None and item.assigned_advantage != 0
        for item in report.learning
    )
    assert all(
        item.anchored is not None
        for item in report.learning
        if item.ship_step is not None
    )
    assert any(
        item.ship_step is not None
        and item.policy_start is not None
        and item.policy_start > 0
        for item in report.learning
    )
    first = training / "checkpoints/step_1/trainer"
    final = training / "checkpoints/step_4/trainer"
    assert checkpoint_sha256(first) != checkpoint_sha256(final)
    run.run_experiment(
        directory,
        RunRequest(
            RunStage.EVALUATE,
            number=int(output.name),
            checkpoint=final,
            split=TaskRole.VALIDATION,
        ),
    )
    evaluation = EvaluationReport.model_validate_json(
        (output / "evaluation/validation/000/evaluation.json").read_bytes()
    )
    assert evaluation.model.adapter_path == final.parent / "adapter"
    assert evaluation.local_server is not None
    assert evaluation.local_server.model.root == str(final.parent / "adapter")
    assert (
        evaluation.local_server.model.parent == evaluation.local_server.context_model.id
    )
    assert evaluation.local_server.context_model.root == str(base)
    assert evaluation.summary.recorded_rollout_count == 1
    assert evaluation.summary.performance.failure_count == 0
    first_model = checkpoint_model(config.model, training, first)
    assert first_model.adapter_path is not None
    assert sha256_file(
        first_model.adapter_path / "adapter_model.safetensors"
    ) != sha256_file(final.parent / "adapter/adapter_model.safetensors")
    first_tensors = load(
        (first_model.adapter_path / "adapter_model.safetensors").read_bytes()
    )
    final_tensors = load(
        (final.parent / "adapter/adapter_model.safetensors").read_bytes()
    )
    assert first_tensors.keys() == final_tensors.keys()
    assert any(
        not torch.equal(first_tensors[name], final_tensors[name])
        for name in first_tensors
    )
    # The evaluator records the preflight-verified advertised base/adapter identity.
    assert (output / "evaluation/validation/000/evaluation.json").is_file()
