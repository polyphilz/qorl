"""Opt-in real policy rollouts, optimizer updates, weight handoff and evaluation."""

import json
import os
from pathlib import Path
from typing import Protocol

import pytest
import tomli_w
import torch
import verifiers.v1 as vf
from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces import update as native_updates
from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.orchestrator import trajectories
from prime_rl.transports.batch import TrainingSample
from pydantic import BaseModel, TypeAdapter
from safetensors.torch import load
from verifiers.v1.graph import tools_hash

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
from qorl.model.client import JSON_OBJECT
from qorl.model.schemas import JsonObject, ToolDefinition
from qorl.rl.report import RlTrainingReport
from qorl.rl.schemas import RlRolloutRecord
from qorl.rl.tasks import QorlTaskData
from qorl.rl.train import checkpoint_model
from qorl.taskset.schemas import TaskRole
from qorl.util.hashing import sha256_file

pytestmark = pytest.mark.skipif(
    os.environ.get("QORL_TEST_RL_GPU") != "1",
    reason="set QORL_TEST_RL_GPU=1 in the agreed isolated benchmark-host checkout",
)


class RecordedEpisode(BaseModel):
    traces: list[vf.Trace[QorlTaskData, vf.State, vf.AgentConfig]]


class NativeUpdates(Protocol):
    def fold_trace_updates(
        self, trace: JsonObject, updates: list[JsonObject]
    ) -> int: ...


class NativeTrajectories(Protocol):
    def trace_to_samples(
        self, trace: vf.Trace[QorlTaskData, vf.State, vf.AgentConfig]
    ) -> list[TrainingSample]: ...


def verify_native_rollouts(
    training: Path,
    annotation_api: NativeUpdates = native_updates,
    trajectory_api: NativeTrajectories = trajectories,
) -> None:
    """Replay retained arrival and ship evidence without generation or database work."""
    report = RlTrainingReport.model_validate_json(
        (training / "report.json").read_bytes()
    )
    credits = {item.trace_id: item.assigned_advantage for item in report.learning}
    updates: dict[str, list[JsonObject]] = {}
    for producer in sorted(get_annotations_dir(training).glob("*")):
        if not producer.is_dir():
            continue
        for chunk in sorted(chunk_numbers(producer)):
            with open_chunk(producer, chunk) as stream:
                for line in stream:
                    update = JSON_OBJECT.validate_json(line)
                    identifier = update["trace_id"]
                    assert isinstance(identifier, str)
                    updates.setdefault(identifier, []).append(update)
    checked_turns = multi_attempt_rollouts = sampled_tokens = context_tokens = (
        credited_tokens
    ) = 0
    stream_path = get_trace_stream(training)
    for chunk in chunk_numbers(stream_path):
        with open_chunk(stream_path, chunk) as stream:
            for line in stream:
                episode = RecordedEpisode.model_validate_json(line)
                for trace in episode.traces:
                    if trace.info.get("qorl_policy") is None:
                        continue
                    policy = AgentTrace.model_validate(trace.info["qorl_policy"])
                    record = RlRolloutRecord.model_validate_json(
                        json.dumps(trace.info["qorl"])
                    )
                    multi_attempt_rollouts += len(record.candidates) > 1
                    assert record.execution_counts is not None
                    if record.final is not None:
                        assert record.execution_counts.initial_default >= 2
                    for node in trace.nodes:
                        assert len(node.mask) == len(node.token_ids)
                        assert len(node.logprobs) == sum(node.mask)
                        if not node.sampled:
                            assert not any(node.mask)
                        sampled_tokens += sum(node.mask)
                        context_tokens += len(node.mask) - sum(node.mask)
                    calls = [call for call in trace.calls if call.node is not None]
                    if record.final is not None:
                        assert len(calls) == len(policy.model_responses)
                    for call, response in zip(
                        calls, policy.model_responses, strict=False
                    ):
                        assert response.prompt_tokens == response.usage.prompt_tokens
                        assert response.request["tool_choice"] == "auto"
                        assert response.request["parallel_tool_calls"] is True
                        tools = TypeAdapter(list[ToolDefinition]).validate_python(
                            response.request["tools"]
                        )
                        assert all(tool in policy.tools for tool in tools)
                        assert policy.agent_interface_version == 6
                        assert call.node is not None
                        assert trace.nodes[call.node].tools_hash == tools_hash(
                            [
                                vf.Tool(
                                    name=tool.function.name,
                                    description=tool.function.description,
                                    parameters=tool.function.parameters,
                                )
                                for tool in tools
                            ]
                        )
                        checked_turns += 1
                    if credits.get(trace.id) is not None:
                        wire = JSON_OBJECT.validate_json(trace.model_dump_json())
                        annotation_api.fold_trace_updates(
                            wire, updates.get(trace.id, [])
                        )
                        folded = type(trace).model_validate_json(json.dumps(wire))
                        trainable_nodes = {
                            id(node)
                            for branch in folded.branches
                            if branch.trainable
                            for node in branch.nodes
                        }
                        expected_count = sum(
                            sum(node.mask)
                            for node in folded.nodes
                            if id(node) in trainable_nodes
                        )
                        actual_count = 0
                        for sample in trajectory_api.trace_to_samples(folded):
                            assert sample.advantages is not None
                            for mask, credit in zip(
                                sample.mask, sample.advantages, strict=True
                            ):
                                if mask:
                                    assert credit == pytest.approx(credits[trace.id])
                                    credited_tokens += 1
                                    actual_count += 1
                        assert actual_count == expected_count
    assert checked_turns > 0 and multi_attempt_rollouts > 0
    assert sampled_tokens > 0 and context_tokens > 0 and credited_tokens > 0
    print(
        f"Native evidence: {checked_turns} turns, {multi_attempt_rollouts} multi-attempt rollouts, "
        f"{sampled_tokens} sampled / {context_tokens} context tokens, {credited_tokens} credited transport tokens"
    )


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
    document["agent"].update(candidate_attempts=5, maximum_model_turns=8)
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
    verify_native_rollouts(training)
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
