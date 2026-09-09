"""Offline preparation against real renderers and Prime-RL's CPU packer."""

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import tomli_w
import torch
import verifiers.v1 as vf
from datasets import Dataset
from openai import AsyncOpenAI
from prime_rl.orchestrator import trajectories
from prime_rl.orchestrator.algo import routing
from prime_rl.trainer.sft.data import Sample, SFTDataset, cat_collate
from prime_rl.transports.batch import TrainingSample
from pydantic import TypeAdapter, ValidationError
from renderers.configs import (
    AutoRendererConfig,
    Qwen3RendererConfig,
    Qwen35RendererConfig,
    Qwen38RendererConfig,
    RendererConfig,
)
from tests.qorl.agent.test_agent import ScriptedTransport, policy
from tests.qorl.measure.test_feedback import ENABLED, InspectionWorker
from tests.qorl.measure.test_rollout import ACTION, TASK, Sql
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifiers.v1 import graph
from verifiers.v1.clients import train
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import Message as NativeMessage
from verifiers.v1.types import SamplingConfig

from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor
from qorl.experiment import create, run
from qorl.experiment.schemas import (
    CreateRequest,
    ExperimentMethod,
    RunRequest,
    RunStage,
    SftExperimentConfig,
    load_config,
)
from qorl.measure.rollout import RolloutEvaluator
from qorl.model.schemas import (
    FunctionCall,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolFunction,
)
from qorl.sft import dataset
from qorl.sft.schemas import (
    Conversation,
    ConversationRequest,
    DatasetPreparationReport,
    JsonObject,
    PackingRowMetadata,
    PreparedDatasetManifest,
    PreparedDatasetSplit,
    RenderedConversation,
    RenderingRejection,
    TrainingRow,
    load_json_lines,
)
from qorl.taskset.schemas import BenchmarkId, TaskRole, TaskSelection
from qorl.taskset.selection import select_tasks
from qorl.taskset.taskset import TaskSet

CONTEXT_LENGTH = 4096
FULL_TOOL_CONTEXT_LENGTH = 32_768
SOURCE_SEED = 17
GENERATION_SEED = 23
SHUFFLE_SEED = 42
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


class ByteAlphabet(Protocol):
    """The byte-level pretokenizer's complete alphabet for the test vocabulary."""

    def alphabet(self) -> list[str]: ...


class SavedTokenizer(dataset.StudentTokenizer, Protocol):
    """The test tokenizer's vocabulary-editing and save operations."""

    def add_tokens(self, new_tokens: list[str]) -> int: ...

    def save_pretrained(self, save_directory: Path) -> tuple[str, ...]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class DatasetLoader(Protocol):
    """JSON records accepted by Hugging Face's in-memory dataset constructor."""

    def from_list(self, mapping: list[JsonObject]) -> Dataset: ...


BYTE_ALPHABET: ByteAlphabet = pre_tokenizers.ByteLevel
DATASET_LOADER: DatasetLoader = Dataset

type NativeTrace = vf.Trace[vf.TaskData, vf.State, vf.AgentConfig]


class CreditRouting(Protocol):
    def assign_advantages(self, trace: NativeTrace, values: float) -> None: ...


class Trajectories(Protocol):
    def trace_to_samples(self, trace: NativeTrace) -> list[TrainingSample]: ...


class Graph(Protocol):
    def prepare_turn(
        self, trace: NativeTrace, prompt: list[NativeMessage]
    ) -> graph.PendingTurn: ...


class NativeTrainingClient(Protocol):
    """Typed signatures for the pinned native client's chat-only test boundary."""

    client: AsyncOpenAI

    async def relay_aux(
        self,
        dialect: ChatDialect,
        route: str,
        body: JsonObject,
        *,
        sampling: SamplingConfig,
        turn: graph.PendingTurn,
    ) -> JsonObject: ...

    async def get_response(
        self,
        dialect: ChatDialect,
        body: JsonObject,
        sampling: SamplingConfig,
        *,
        turn: graph.PendingTurn,
    ) -> vf.Response: ...

    async def close(self) -> None: ...


def native_training_client() -> NativeTrainingClient:
    return train.TrainClient(TrainClientConfig.model_validate({"api_key_var": ""}))


@pytest.fixture
def tokenizer() -> SavedTokenizer:
    """A byte tokenizer with real offsets and Qwen delimiters; no downloaded weights."""
    tokens = [*SPECIAL_TOKENS, *sorted(BYTE_ALPHABET.alphabet())]
    backend = Tokenizer(
        models.BPE({token: index for index, token in enumerate(tokens)}, [])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=SPECIAL_TOKENS,
        eos_token="<|im_end|>",
    )


@pytest.fixture
def student_model(tmp_path: Path, tokenizer: SavedTokenizer) -> Path:
    """Save the test tokenizer beside placeholder weights that must not be loaded."""
    path = tmp_path / "student"
    tokenizer.save_pretrained(path)
    (path / "config.json").write_text('{"model_type":"qwen3"}')
    # Preparation checks identity/file presence, but must never load these tensors.
    (path / "model.safetensors").write_bytes(b"not tensors")
    return path


@pytest.fixture
def tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            function=ToolFunction(
                name="inspect",
                description="Inspect a relation",
                parameters={
                    "type": "object",
                    "properties": {"alias": {"type": "string"}},
                },
            )
        )
    ]


def conversation(
    task_id: str, identifier: str, tools: list[ToolDefinition]
) -> Conversation:
    return Conversation(
        conversation_id=identifier,
        task_id=task_id,
        metadata={"generator": "frozen-teacher", "attempt": 1},
        requests=[
            ConversationRequest(assistant_message_index=2, tools=tools),
            ConversationRequest(assistant_message_index=4, tools=[]),
        ],
        messages=[
            Message(role=MessageRole.SYSTEM, content="SYSTEM CONTEXT"),
            Message(role=MessageRole.USER, content="QUERY CONTEXT"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Inspect first.",
                tool_calls=[
                    ToolCall(
                        id="call",
                        function=FunctionCall(
                            name="inspect", arguments='{"alias":"t"}'
                        ),
                    )
                ],
            ),
            Message(role=MessageRole.TOOL, tool_call_id="call", content="TOOL CONTEXT"),
            Message(role=MessageRole.ASSISTANT, content="Done."),
        ],
    )


@pytest.fixture
def experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    student_model: Path,
    tools: list[ToolDefinition],
    benchmark_task_sets: dict[str, TaskSet],
) -> Path:
    monkeypatch.setattr(create, "EXPERIMENTS_DIRECTORY", tmp_path / "experiments")
    monkeypatch.setattr(run, "OUTPUTS_DIRECTORY", tmp_path / "outputs")
    source = tmp_path / "source"
    source.mkdir()
    selected = select_tasks(
        ["train=ceb[2a:1]", "validation=ceb[4a:1]"], benchmark_task_sets, SOURCE_SEED
    )
    training = [
        conversation(selected[TaskRole.TRAIN].task_ids[0], f"train-{index}", tools)
        for index in range(2)
    ]
    validation = [
        conversation(selected[TaskRole.VALIDATION].task_ids[0], "validation", tools)
    ]
    dataset.write_records(source / "training.jsonl", training)
    dataset.write_records(source / "validation.jsonl", validation)
    manifest = PreparedDatasetManifest(
        schema_version=2,
        format="qorl-conversations",
        training=PreparedDatasetSplit(
            selection=selected[TaskRole.TRAIN],
            selection_seed=SOURCE_SEED,
            generation_seed=GENERATION_SEED,
            conversations=Path("training.jsonl"),
        ),
        validation=PreparedDatasetSplit(
            selection=selected[TaskRole.VALIDATION],
            selection_seed=SOURCE_SEED,
            generation_seed=GENERATION_SEED,
            conversations=Path("validation.jsonl"),
        ),
    )
    (source / "manifest.json").write_text(manifest.model_dump_json())
    directory = create.create_experiment(
        CreateRequest(
            name="reused",
            method=ExperimentMethod.SFT,
            seed=SHUFFLE_SEED,
            base_model_name_or_path=str(student_model),
            dataset_from=source,
            postgres_config=Path("docker/postgres/configs/000-pgconf-default"),
            pool_config=Path("docker/worker_pool/configs/002-poolconf-4x8"),
        )
    )
    config = sft_config(directory)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"context_length": CONTEXT_LENGTH}),
            "inference": config.inference.model_copy(update={"max_tokens": 2048}),
            "training": config.training.model_copy(
                update={"renderer": Qwen3RendererConfig()}
            ),
        }
    )
    (directory / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    return directory


def sft_config(directory: Path) -> SftExperimentConfig:
    config = load_config(directory / "config.toml")
    assert isinstance(config, SftExperimentConfig)
    return config


@pytest.mark.parametrize(
    "renderer_config,argument_text",
    [
        (Qwen3RendererConfig(), '"t"'),
        (Qwen35RendererConfig(), "\nt\n"),
        (Qwen38RendererConfig(), "\nt\n"),
    ],
)
@pytest.mark.parametrize("thinking", [False, True])
def test_masks_and_targets_match_native_sft_processing(
    experiment: Path,
    renderer_config: RendererConfig,
    argument_text: str,
    thinking: bool,
) -> None:
    config = sft_config(experiment)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"renderer": renderer_config}
            ),
            "inference": config.inference.model_copy(update={"thinking": thinking}),
        }
    )
    inputs = run.load_inputs(experiment)
    source = dataset.load_source(config, inputs.selections)
    student = dataset.load_student(config)
    record = source.conversations[TaskRole.TRAIN][0]
    rendered = dataset.render_conversation(record, student, CONTEXT_LENGTH)
    assert rendered.rejection is None
    # Prime-RL owns mask construction; compare QORL's actual shifted tensors.
    targets_by_message: dict[int, list[int]] = {}
    for request, sample in zip(record.requests, rendered.requests, strict=True):
        native = SFTDataset(
            DATASET_LOADER.from_list(
                [
                    {
                        "messages": [
                            message.model_dump(mode="json", exclude_none=True)
                            for message in record.messages[
                                : request.assistant_message_index + 1
                            ]
                        ],
                        "tools": json.dumps(
                            [tool.model_dump(mode="json") for tool in request.tools]
                        ),
                    }
                ]
            ),
            student.renderer,
            shuffle=False,
            seq_len=CONTEXT_LENGTH,
            max_epochs=1,
        )
        expected = next(dataset.training_rows(native))
        expected = expected.model_copy(
            update={
                "loss_mask": [
                    mask and index == request.assistant_message_index
                    for mask, index in zip(
                        expected.loss_mask, sample.target_message_indices, strict=True
                    )
                ]
            }
        )
        assert sample.sample == expected
        for target, trainable, message_index in zip(
            sample.sample.target_ids,
            sample.sample.loss_mask,
            sample.target_message_indices,
            strict=True,
        ):
            if trainable:
                assert message_index == request.assistant_message_index
                targets_by_message.setdefault(message_index, []).append(target)
    assert set(targets_by_message) == {2, 4}
    tool_turn = student.tokenizer.decode(
        targets_by_message[2], skip_special_tokens=False
    )
    assert (
        "<tool_call>" in tool_turn
        and "inspect" in tool_turn
        and argument_text in tool_turn
    )
    assert all(
        targets[-1] in student.renderer.get_stop_token_ids()
        for targets in targets_by_message.values()
    )
    assert "SYSTEM CONTEXT" in rendered.requests[-1].rendered_text
    assert "TOOL CONTEXT" in rendered.requests[-1].rendered_text


def test_retired_tools_and_exact_context_are_preserved_in_saved_conversations(
    experiment: Path, tools: list[ToolDefinition]
) -> None:
    legacy = tools[0].model_copy(
        update={
            "function": tools[0].function.model_copy(update={"name": "describe_table"})
        }
    )
    record = conversation("legacy-task", "legacy-interface", [legacy])
    record.messages[2] = record.messages[2].model_copy(
        update={
            "tool_calls": [
                ToolCall(
                    id="call",
                    function=FunctionCall(
                        name="describe_table", arguments='{"alias":"t"}'
                    ),
                ),
            ]
        }
    )
    original = record.model_dump_json()
    restored = Conversation.model_validate_json(original)
    student = dataset.load_student(sft_config(experiment))
    rendered = dataset.render_conversation(restored, student, CONTEXT_LENGTH)
    assert rendered.rejection is None
    assert restored.model_dump_json() == original
    assert "describe_table" in rendered.requests[0].rendered_text
    assert "inspect_relation" not in rendered.requests[0].rendered_text
    assert restored.requests[0].tools == [legacy]
    assert restored.requests[1].tools == []
    assert "SYSTEM CONTEXT" in rendered.requests[1].rendered_text
    assert "QUERY CONTEXT" in rendered.requests[1].rendered_text
    assert "TOOL CONTEXT" in rendered.requests[1].rendered_text


def test_two_conversations_have_independent_targets_and_positions(
    experiment: Path,
) -> None:
    config = sft_config(experiment)
    source = dataset.load_source(config, run.load_inputs(experiment).selections)
    student = dataset.load_student(config)
    records = [
        dataset.render_conversation(item, student, CONTEXT_LENGTH)
        for item in source.conversations[TaskRole.TRAIN]
    ]
    rows, metadata = dataset.pack_conversations(records, CONTEXT_LENGTH, SHUFFLE_SEED)
    assert len(rows) == 1
    assert len(rows[0].seq_lens) == 4
    by_id = {
        (request.conversation_id, request.assistant_message_index): request
        for record in records
        for request in record.requests
    }
    offset = 0
    for identifier, message_index, length in zip(
        metadata[0].conversation_ids,
        metadata[0].assistant_message_indices,
        metadata[0].request_lengths,
        strict=True,
    ):
        sample = by_id[identifier, message_index].sample
        assert rows[0].input_ids[offset : offset + length] == sample.input_ids
        assert rows[0].target_ids[offset : offset + length] == sample.target_ids
        assert rows[0].loss_mask[offset : offset + length] == sample.loss_mask
        assert rows[0].position_ids[offset : offset + length] == list(range(length))
        offset += length
    assert metadata[0].padding_tokens == CONTEXT_LENGTH - offset
    assert not any(rows[0].loss_mask[offset:])
    batch = cat_collate([Sample(**rows[0].model_dump())])
    assert torch.equal(batch["seq_lens"], torch.tensor(rows[0].seq_lens))
    assert dataset.pack_conversations(records, CONTEXT_LENGTH, SHUFFLE_SEED) == (
        rows,
        metadata,
    )


def test_prepare_is_offline_preserves_sources_and_resumes(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    original = {
        path.name: path.read_bytes() for path in config.data.dataset_from.iterdir()
    }
    monkeypatch.setattr(
        run.PostgresConfig,
        "load",
        Mock(side_effect=AssertionError("no PostgreSQL during preparation")),
    )
    output = run.run_experiment(experiment, RunRequest(RunStage.PREPARE))
    directory = output / "dataset"
    report = DatasetPreparationReport.model_validate_json(
        (directory / "report.json").read_bytes()
    )
    assert (
        report.training.source_conversations
        == report.training.accepted_conversations
        == 2
    )
    assert report.training.packed_rows == 1
    assert report.training.accepted_requests == 4
    assert report.validation.accepted_requests == 2
    assert (
        report.validation.accepted_conversations == report.validation.packed_rows == 1
    )
    assert {
        path.name: path.read_bytes() for path in config.data.dataset_from.iterdir()
    } == original
    for filename in ("training.jsonl", "validation.jsonl"):
        assert (directory / filename).read_bytes() == original[filename]
    frozen = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        dataset,
        "render_conversation",
        Mock(side_effect=AssertionError("must not render completed work")),
    )
    monkeypatch.setattr(
        dataset,
        "pack_conversations",
        Mock(side_effect=AssertionError("must not repack completed work")),
    )
    assert (
        run.run_experiment(
            experiment, RunRequest(RunStage.PREPARE, number=0, resume=True)
        )
        == output
    )
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in frozen
    } == frozen
    with pytest.raises(ValueError, match="already exists"):
        run.run_experiment(experiment, RunRequest(RunStage.PREPARE, number=0))
    reused = dataset.load_source(
        config.model_copy(
            update={"data": config.data.model_copy(update={"dataset_from": directory})}
        ),
        run.load_inputs(experiment).selections,
    )
    assert reused.manifest.training.selection_seed == SOURCE_SEED
    assert reused.manifest.training.generation_seed == GENERATION_SEED


def test_partial_resume_keeps_completed_renders(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render = dataset.render_conversation
    interrupted = Mock(side_effect=[render, RuntimeError("interrupted")])

    # Wrap the real renderer: fail on the second conversation only.
    def fail_second(
        record: Conversation,
        student: dataset.StudentRenderer,
        context_length: int,
    ) -> RenderedConversation:
        result = interrupted()
        return result(record, student, context_length)

    monkeypatch.setattr(dataset, "render_conversation", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_experiment(experiment, RunRequest(RunStage.PREPARE))
    completed = (
        run.OUTPUTS_DIRECTORY
        / experiment.name
        / "000/dataset/training/rendered/000000.json"
    )
    original = completed.read_bytes(), completed.stat().st_mtime_ns
    continued = Mock(wraps=render)
    monkeypatch.setattr(dataset, "render_conversation", continued)
    run.run_experiment(experiment, RunRequest(RunStage.PREPARE, number=0, resume=True))
    assert continued.call_count == 2
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == original


@pytest.mark.parametrize("change", ["source", "tools", "config", "prepared"])
def test_resume_rejects_changed_inputs(experiment: Path, change: str) -> None:
    output = run.run_experiment(experiment, RunRequest(RunStage.PREPARE))
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    if change == "source":
        path = config.data.dataset_from / "training.jsonl"
        path.write_text(path.read_text() + "\n")
    elif change == "tools":
        path = config.data.dataset_from / "training.jsonl"
        records = load_json_lines(path, Conversation)
        original = records[0].requests[0].tools[0]
        records[0].requests[0].tools[0] = original.model_copy(
            update={
                "function": original.function.model_copy(
                    update={"description": "Changed tool description"}
                )
            }
        )
        dataset.write_records(path, records)
    elif change == "config":
        path = experiment / "config.toml"
        path.write_text(path.read_text().replace("seed = 42", "seed = 43"))
    else:
        path = output / "dataset/training/packed.jsonl"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        run.run_experiment(
            experiment, RunRequest(RunStage.PREPARE, number=0, resume=True)
        )


def test_overlong_rejection_happens_before_native_packing(experiment: Path) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "training.jsonl"
    records = load_json_lines(path, Conversation)
    messages = list(records[-1].messages)
    messages[3] = messages[3].model_copy(update={"content": "x" * CONTEXT_LENGTH})
    long = records[-1].model_copy(update={"messages": messages})
    dataset.write_records(path, [records[0], long])
    output = run.run_experiment(experiment, RunRequest(RunStage.PREPARE)) / "dataset"
    report = DatasetPreparationReport.model_validate_json(
        (output / "report.json").read_bytes()
    )
    assert report.training.source_conversations == 2
    assert report.training.accepted_conversations == 1
    assert report.training.rejections[RenderingRejection.OVER_CONTEXT] == 1
    rejected = RenderedConversation.model_validate_json(
        (output / "training/rendered/000001.json").read_bytes()
    )
    assert rejected.rejection == RenderingRejection.OVER_CONTEXT
    assert len(rejected.requests[0].sample.input_ids) < CONTEXT_LENGTH
    assert len(rejected.requests[1].sample.input_ids) > CONTEXT_LENGTH
    packing = load_json_lines(output / "training/packing.jsonl", PackingRowMetadata)
    assert all(long.conversation_id not in row.conversation_ids for row in packing)
    assert len(load_json_lines(output / "training/packed.jsonl", TrainingRow)) == 1


def test_unknown_local_renderer_requires_explicit_selection(experiment: Path) -> None:
    config = sft_config(experiment)

    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"renderer": AutoRendererConfig()}
            )
        }
    )
    with pytest.raises(ValueError, match=r"set training\.renderer explicitly"):
        dataset.load_student(config)


def test_tokenizer_loader_result_is_checked(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = Mock(from_pretrained=Mock(return_value=object()))
    monkeypatch.setattr(dataset, "TOKENIZER_LOADER", loader)
    with pytest.raises(ValueError, match="requires a fast text tokenizer"):
        dataset.load_student(sft_config(experiment))


def test_native_training_rows_are_validated() -> None:
    with pytest.raises(ValidationError, match="target_ids"):
        list(dataset.training_rows([{"input_ids": [1]}]))


def test_reasoning_is_retained_and_supervised(experiment: Path) -> None:
    config = sft_config(experiment)
    source = dataset.load_source(config, run.load_inputs(experiment).selections)
    student = dataset.load_student(config)
    record = source.conversations[TaskRole.TRAIN][0]
    messages = list(record.messages)
    messages[-1] = messages[-1].model_copy(
        update={"reasoning_content": "VISIBLE REASONING"}
    )
    rendered = dataset.render_conversation(
        record.model_copy(update={"messages": messages}),
        student,
        CONTEXT_LENGTH,
    )
    trained = [
        target
        for target, mask in zip(
            rendered.requests[-1].sample.target_ids,
            rendered.requests[-1].sample.loss_mask,
            strict=True,
        )
        if mask
    ]
    assert "VISIBLE REASONING" in student.tokenizer.decode(trained)


def test_context_boundary_is_the_shifted_sequence_length(experiment: Path) -> None:
    config = sft_config(experiment)
    source = dataset.load_source(config, run.load_inputs(experiment).selections)
    student = dataset.load_student(config)
    record = source.conversations[TaskRole.TRAIN][0]
    record = record.model_copy(
        update={"messages": record.messages[:3], "requests": record.requests[:1]}
    )
    length = len(
        dataset.render_conversation(record, student, CONTEXT_LENGTH)
        .requests[0]
        .sample.input_ids
    )
    accepted = dataset.render_conversation(record, student, length)
    rejected = dataset.render_conversation(record, student, length - 1)
    assert accepted.rejection is None
    assert rejected.rejection == RenderingRejection.OVER_CONTEXT
    rows, metadata = dataset.pack_conversations([accepted], length, SHUFFLE_SEED)
    assert rows == [accepted.requests[0].sample]
    assert metadata[0].padding_tokens == 0
    assert dataset.pack_conversations([rejected], length - 1, SHUFFLE_SEED) == ([], [])


def test_empty_validation_is_reported_and_stays_failed_on_resume(
    experiment: Path,
) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "validation.jsonl"
    record = load_json_lines(path, Conversation)[0]
    record = record.model_copy(
        update={
            "messages": [
                Message(role=MessageRole.USER, content="No assistant targets")
            ],
            "requests": [],
        }
    )
    dataset.write_records(path, [record])
    with pytest.raises(ValueError, match="empty training or validation"):
        run.run_experiment(experiment, RunRequest(RunStage.PREPARE))
    report = DatasetPreparationReport.model_validate_json(
        (
            run.OUTPUTS_DIRECTORY / experiment.name / "000/dataset/report.json"
        ).read_bytes()
    )
    assert report.validation.rejections[RenderingRejection.NO_TARGETS] == 1
    assert report.validation.packed_rows == 0
    with pytest.raises(ValueError, match="empty training or validation"):
        run.run_experiment(
            experiment, RunRequest(RunStage.PREPARE, number=0, resume=True)
        )


@pytest.mark.parametrize("field", ["selection_seed", "generation_seed"])
def test_changed_source_seeds_do_not_get_reinterpreted(
    experiment: Path, field: str
) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "manifest.json"
    manifest = PreparedDatasetManifest.model_validate_json(path.read_bytes())
    manifest = manifest.model_copy(
        update={"training": manifest.training.model_copy(update={field: SHUFFLE_SEED})}
    )
    path.write_text(manifest.model_dump_json())
    with pytest.raises(ValueError, match="selection or seeds changed"):
        dataset.load_source(config, run.load_inputs(experiment).selections)


def test_source_files_cannot_escape_the_artifact(experiment: Path) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "manifest.json"
    manifest = PreparedDatasetManifest.model_validate_json(path.read_bytes())
    path.write_text(
        manifest.model_copy(
            update={
                "training": manifest.training.model_copy(
                    update={"conversations": Path("../training.jsonl")}
                )
            }
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="inside its artifact"):
        dataset.load_source(config, run.load_inputs(experiment).selections)


def test_prepare_entrypoint_reports_success(
    experiment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run.main(experiment, ["--stage", "prepare"]) == 0
    assert "Prepared 2 training conversations in 1 rows" in capsys.readouterr().out


def test_tokenizer_change_prevents_resume(
    experiment: Path, student_model: Path, tokenizer: SavedTokenizer
) -> None:
    run.run_experiment(experiment, RunRequest(RunStage.PREPARE))
    tokenizer.add_tokens(["CHANGED_TOKENIZER"])
    tokenizer.save_pretrained(student_model)
    with pytest.raises(ValueError, match="inputs changed"):
        run.run_experiment(
            experiment, RunRequest(RunStage.PREPARE, number=0, resume=True)
        )


@pytest.mark.parametrize("problem", ["task", "id", "tool", "arguments", "orphan"])
def test_invalid_source_fails_before_tokenization(
    experiment: Path, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "training.jsonl"
    records = load_json_lines(path, Conversation)
    if problem == "task":
        records[0] = records[0].model_copy(update={"task_id": "not-selected"})
    elif problem == "id":
        records[1] = records[1].model_copy(
            update={"conversation_id": records[0].conversation_id}
        )
    elif problem == "orphan":
        records[0] = records[0].model_copy(
            update={
                "messages": [
                    Message(
                        role=MessageRole.TOOL, content="result", tool_call_id="missing"
                    )
                ],
                "requests": [],
            }
        )
    else:
        call = ToolCall(
            id="call",
            function=FunctionCall(
                name="missing" if problem == "tool" else "inspect",
                arguments="[]" if problem == "arguments" else "{}",
            ),
        )
        records[0] = records[0].model_copy(
            update={
                "messages": [Message(role=MessageRole.ASSISTANT, tool_calls=[call])],
                "requests": [
                    records[0]
                    .requests[0]
                    .model_copy(update={"assistant_message_index": 0})
                ],
            }
        )
    dataset.write_records(path, records)
    monkeypatch.setattr(
        dataset,
        "load_student",
        Mock(side_effect=AssertionError("must reject source first")),
    )
    with pytest.raises(ValueError):
        run.run_experiment(experiment, RunRequest(RunStage.PREPARE))


def test_job_queries_render_their_own_alias_schemas(
    experiment: Path, benchmark_task_sets: dict[str, TaskSet]
) -> None:
    config = sft_config(experiment)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"context_length": FULL_TOOL_CONTEXT_LENGTH}
            )
        }
    )
    (experiment / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )
    assert config.data.dataset_from is not None
    manifest_path = config.data.dataset_from / "manifest.json"
    manifest = PreparedDatasetManifest.model_validate_json(manifest_path.read_bytes())
    for split, task_id, alias in (
        ("training", "job-01a", "ct"),
        ("validation", "job-02a", "cn"),
    ):
        task = next(
            task for task in benchmark_task_sets["job"].tasks if task.task_id == task_id
        )
        tools = agent_tools(
            [relation.alias for relation in task.relations], execution_feedback=False
        )
        record = Conversation(
            conversation_id=task_id,
            task_id=task_id,
            metadata={},
            messages=[
                Message(role=MessageRole.USER, content=f"Inspect {alias}"),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="inspect",
                            function=FunctionCall(
                                name="inspect_relation",
                                arguments=json.dumps({"relation": alias}),
                            ),
                        )
                    ],
                ),
            ],
            requests=[ConversationRequest(assistant_message_index=1, tools=tools)],
        )
        selection = TaskSelection(benchmark_id=BenchmarkId.JOB, task_ids=[task_id])
        (experiment / f"{split}-tasks.json").write_text(selection.model_dump_json())
        source_split = manifest.training if split == "training" else manifest.validation
        manifest = manifest.model_copy(
            update={split: source_split.model_copy(update={"selection": selection})}
        )
        dataset.write_records(config.data.dataset_from / f"{split}.jsonl", [record])
    manifest_path.write_text(manifest.model_dump_json())
    output = run.run_experiment(experiment, RunRequest(RunStage.PREPARE)) / "dataset"
    for split, alias, excluded_alias in (
        ("training", "ct", "cn"),
        ("validation", "cn", "ct"),
    ):
        rendered = RenderedConversation.model_validate_json(
            (output / split / "rendered/000000.json").read_bytes()
        )
        text = rendered.requests[0].rendered_text
        schema_text = text.split("<tools>\n")[1].split("\n</tools>")[0]
        actual = [
            ToolDefinition.model_validate_json(line)
            for line in schema_text.splitlines()
        ]
        original = load_json_lines(
            config.data.dataset_from / f"{split}.jsonl", Conversation
        )[0]
        assert actual == original.requests[0].tools
        describe = next(
            tool for tool in actual if tool.function.name == "inspect_relation"
        )
        encoded = describe.model_dump_json()
        assert f'"{alias}"' in encoded and f'"{excluded_alias}"' not in encoded


def test_each_reply_uses_only_its_own_request_tools(experiment: Path) -> None:
    config = sft_config(experiment)
    source = dataset.load_source(config, run.load_inputs(experiment).selections)
    record = source.conversations[TaskRole.TRAIN][0]
    finish = ToolDefinition(
        function=ToolFunction(
            name="finish",
            description="Finish",
            parameters={"type": "object", "additionalProperties": False},
        )
    )
    messages = list(record.messages)
    messages[-1] = Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[
            ToolCall(
                id="finish-call", function=FunctionCall(name="finish", arguments="{}")
            )
        ],
    )
    requests = [
        record.requests[0],
        ConversationRequest(assistant_message_index=4, tools=[finish]),
    ]
    record = record.model_copy(update={"messages": messages, "requests": requests})
    dataset.validate_conversation(record)
    student = dataset.load_student(config)
    rendered = dataset.render_conversation(record, student, CONTEXT_LENGTH)
    first, last = rendered.requests
    for request, expected in ((first, record.requests[0]), (last, record.requests[1])):
        schema_text = request.rendered_text.split("<tools>\n")[1].split("\n</tools>")[0]
        assert [
            ToolDefinition.model_validate_json(line)
            for line in schema_text.splitlines()
        ] == expected.tools
    assert first.tools_sha256 != last.tools_sha256
    trained = [
        target
        for target, mask in zip(
            last.sample.target_ids, last.sample.loss_mask, strict=True
        )
        if mask
    ]
    text = student.tokenizer.decode(trained)
    assert "finish" in text and "inspect" not in text and "TOOL CONTEXT" not in text
    invalid = record.model_copy(
        update={"requests": [requests[0], requests[1].model_copy(update={"tools": []})]}
    )
    with pytest.raises(ValueError, match="unavailable tool finish"):
        dataset.validate_conversation(invalid)


def test_wrong_query_alias_is_rejected_by_the_recorded_schema(
    tools: list[ToolDefinition],
) -> None:
    record = conversation("job-02a", "bad-schema", tools)
    describe = ToolDefinition.model_validate(
        next(
            tool
            for tool in agent_tools(
                ["ct", "it", "mc", "mi_idx", "t"], execution_feedback=False
            )
            if tool.function.name == "inspect_relation"
        )
    )
    messages = list(record.messages)
    messages[2] = Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[
            ToolCall(
                id="call",
                function=FunctionCall(
                    name="inspect_relation", arguments='{"relation":"cn"}'
                ),
            )
        ],
    )
    record = record.model_copy(
        update={
            "messages": messages,
            "requests": [
                ConversationRequest(assistant_message_index=2, tools=[describe]),
                record.requests[1],
            ],
        }
    )
    with pytest.raises(ValueError, match=r"cn.*not one of"):
        dataset.validate_conversation(record)


@pytest.mark.parametrize(
    "next_role", [MessageRole.ASSISTANT, MessageRole.USER, MessageRole.SYSTEM]
)
@pytest.mark.parametrize("delayed_result", [False, True])
def test_unanswered_calls_prevent_advancing_history(
    tools: list[ToolDefinition], next_role: MessageRole, delayed_result: bool
) -> None:
    record = conversation("task", "unanswered", tools)
    messages = [
        *record.messages[:3],
        Message(role=next_role, content="Premature decision"),
    ]
    requests = list(record.requests[:1])
    if next_role == MessageRole.ASSISTANT:
        requests.append(ConversationRequest(assistant_message_index=3, tools=[]))
    if delayed_result:
        messages.append(record.messages[3])
    record = record.model_copy(update={"messages": messages, "requests": requests})
    with pytest.raises(ValueError, match="outstanding tool calls"):
        dataset.validate_conversation(record)


def test_every_outstanding_call_requires_its_own_result(
    tools: list[ToolDefinition],
) -> None:
    record = conversation("task", "two-calls", tools)
    messages = list(record.messages)
    calls = messages[2].tool_calls
    assert calls is not None
    messages[2] = messages[2].model_copy(
        update={"tool_calls": [*calls, calls[0].model_copy(update={"id": "second"})]}
    )
    with pytest.raises(ValueError, match="outstanding tool calls"):
        dataset.validate_conversation(record.model_copy(update={"messages": messages}))
    messages.insert(
        4,
        Message(role=MessageRole.TOOL, tool_call_id="second", content="second result"),
    )
    record = record.model_copy(
        update={
            "messages": messages,
            "requests": [
                record.requests[0],
                ConversationRequest(assistant_message_index=5, tools=[]),
            ],
        }
    )
    dataset.validate_conversation(record)


def test_a_conversation_can_end_at_a_tool_call(tools: list[ToolDefinition]) -> None:
    record = conversation("task", "terminal", tools)
    dataset.validate_conversation(
        record.model_copy(
            update={"messages": record.messages[:3], "requests": record.requests[:1]}
        )
    )


def test_feedback_repair_and_earlier_selection_prepare_exact_context(
    experiment: Path, repository_root: Path
) -> None:
    worker = InspectionWorker(repository_root)
    evaluator = RolloutEvaluator[InspectionExecutor](
        worker, Sql(), TASK, measurement=ENABLED, max_candidates=3
    )
    evaluator.start()
    calls: list[tuple[str, JsonObject]] = [
        ("evaluate_candidate", {"action": ACTION}),
        (
            "evaluate_candidate",
            {
                "action": {
                    "version": 1,
                    "joins": [{"relations": ["a", "b"], "force": "nestloop"}],
                }
            },
        ),
        (
            "evaluate_candidate",
            {"action": {"version": 1, "settings": {"seq_page_cost": 3.0}}},
        ),
        ("finish", {"selected_candidate_id": "candidate-01"}),
    ]
    responses: list[JsonObject] = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": f"Decision {index}",
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
        for index, (name, arguments) in enumerate(calls)
    ]
    transport = ScriptedTransport(responses)
    trace = policy(transport, attempts=3, turns=8).search(evaluator)
    assert evaluator.candidates[0].execution_feedback is not None
    assert not evaluator.candidates[1].constraints_satisfied
    assert evaluator.candidates[2].constraints_satisfied
    assert trace.selection.selected_candidate_id == "candidate-01"
    # Final pairs happen after the last model request and cannot become its context.
    frozen_requests = json.dumps(transport.requests)
    worker.default_ms = 123.456  # A later final-pair timing, absent from all requests.
    evaluator.finish(
        random.Random(42), selected_candidate_id=trace.selection.selected_candidate_id
    )
    assert json.dumps(transport.requests) == frozen_requests
    config = sft_config(experiment)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"context_length": FULL_TOOL_CONTEXT_LENGTH}
            ),
            "training": config.training.model_copy(
                update={"renderer": Qwen35RendererConfig()}
            ),
        }
    )
    inputs = run.load_inputs(experiment)
    source = dataset.load_source(config, inputs.selections)
    requests: list[ConversationRequest] = []
    for response in trace.model_responses:
        wire_messages = TypeAdapter(list[JsonObject]).validate_python(
            response.request["messages"]
        )
        for message in wire_messages:
            # The local chat wire spells the preserved reasoning field "reasoning".
            if "reasoning" in message:
                message["reasoning_content"] = message.pop("reasoning")
        preceding = TypeAdapter(list[Message]).validate_python(wire_messages)
        assert trace.transcript[: len(preceding)] == preceding
        requests.append(
            ConversationRequest(
                assistant_message_index=len(preceding),
                tools=TypeAdapter(list[ToolDefinition]).validate_python(
                    response.request["tools"]
                ),
            )
        )
    assert [tool.function.name for tool in requests[-1].tools] == ["finish"]
    assert len(requests[0].tools) == 5  # finish is not available before candidates.
    assert config.data.dataset_from is not None
    for role, name in (
        (TaskRole.TRAIN, "training"),
        (TaskRole.VALIDATION, "validation"),
    ):
        original = source.conversations[role][0]
        controlled = Conversation(
            conversation_id=original.conversation_id,
            task_id=original.task_id,
            messages=trace.transcript,
            requests=requests,
            metadata={"fixture": "controlled shared-agent feedback; no quality claim"},
        )
        dataset.validate_conversation(controlled)
        dataset.write_records(config.data.dataset_from / f"{name}.jsonl", [controlled])
    frozen_source = (config.data.dataset_from / "training.jsonl").read_bytes()
    controlled = Conversation.model_validate_json(frozen_source)
    output = experiment / "controlled-preparation"
    report = dataset.prepare_dataset(config, inputs.selections, output)
    assert report.training.accepted_conversations == 1
    assert (config.data.dataset_from / "training.jsonl").read_bytes() == frozen_source
    rendered = RenderedConversation.model_validate_json(
        (output / "training/rendered/000000.json").read_bytes()
    )
    student = dataset.load_student(config)
    for index, (request, sample) in enumerate(
        zip(requests, rendered.requests, strict=True)
    ):
        schema_text = sample.rendered_text.split("<tools>\n")[1].split("\n</tools>")[0]
        assert [
            ToolDefinition.model_validate_json(line)
            for line in schema_text.splitlines()
        ] == request.tools
        # Rendering each frozen prefix alone must reproduce the prepared sample.
        prefix = controlled.model_copy(
            update={
                "messages": controlled.messages[: request.assistant_message_index + 1],
                "requests": requests[: index + 1],
            }
        )
        assert (
            dataset.render_conversation(
                prefix, student, FULL_TOOL_CONTEXT_LENGTH
            ).requests[-1]
            == sample
        )
        assert "candidate_median_execution_time_ms" not in sample.rendered_text
        assert "123.456" not in sample.rendered_text
        for target_index, mask in zip(
            sample.target_message_indices, sample.sample.loss_mask, strict=True
        ):
            if mask:
                assert target_index == request.assistant_message_index


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("advantage", [0.4, -0.1, 0.0])
def test_native_renderer_client_credit(
    experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
    advantage: float,
    tokenizer: SavedTokenizer,
) -> None:
    asyncio.run(
        native_renderer_client_credit(
            experiment, monkeypatch, thinking, advantage, tokenizer
        )
    )


async def native_renderer_client_credit(
    experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
    advantage: float,
    tokenizer: SavedTokenizer,
    credit_routing: CreditRouting = routing,
    native_graph: Graph = graph,
    native_trajectories: Trajectories = trajectories,
) -> None:
    """Real native rendering/parsing/transport; only the engine's sampled output is controlled."""
    config = sft_config(experiment)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"renderer": Qwen35RendererConfig()}
            ),
            "inference": config.inference.model_copy(update={"thinking": thinking}),
        }
    )
    student = dataset.load_student(config)
    slot = train.RendererSlot(student.renderer)

    @asynccontextmanager
    async def acquire() -> AsyncGenerator[train.RendererSlot]:
        yield slot

    monkeypatch.setattr(
        train, "ElasticRendererPool", Mock(return_value=Mock(acquire=acquire))
    )
    monkeypatch.setattr(
        "renderers.client._resolve_max_prompt_len",
        AsyncMock(return_value=FULL_TOOL_CONTEXT_LENGTH),
    )
    client = native_training_client()
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="query")),
    )
    messages: list[NativeMessage] = [
        vf.SystemMessage(content="rules"),
        vf.UserMessage(content="query"),
    ]
    expected: list[int] = []
    try:
        for index, name in enumerate(
            ["get_plan", "evaluate_candidate", "evaluate_candidate", "finish"]
        ):
            tools = [
                tool
                for tool in agent_tools(["a", "b"], execution_feedback=True)
                if name != "finish" or tool.function.name == "finish"
            ]
            body: JsonObject = {
                "model": "controlled",
                "tools": [tool.model_dump(mode="json") for tool in tools],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
            turn = native_graph.prepare_turn(trace, messages)
            sampling = SamplingConfig(max_tokens=2048)
            counted = await client.relay_aux(
                ChatDialect(), "/tokenize", body, sampling=sampling, turn=turn
            )
            text = (
                f"reasoning {index}</think>\n" if thinking else ""
            ) + f"<tool_call>\n<function={name}>\n</function>\n</tool_call><|im_end|>"
            completion = tokenizer.encode(text, add_special_tokens=False)
            expected.extend(completion)
            post = AsyncMock(
                return_value=httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "token_ids": completion,
                                "finish_reason": "stop",
                                "logprobs": {
                                    "content": [
                                        {"token": f"token_id:{token}", "logprob": -1.0}
                                        for token in completion
                                    ]
                                },
                            }
                        ]
                    },
                )
            )
            monkeypatch.setattr(client.client, "post", post)
            response = await client.get_response(
                ChatDialect(), body, sampling, turn=turn
            )
            assert response.tokens is not None
            assert counted["count"] == len(response.tokens.prompt_ids)
            assert (
                response.message.tool_calls
                and response.message.tool_calls[0].name == name
            )
            if thinking:
                assert response.message.reasoning_content == f"reasoning {index}"
            turn.commit(
                response,
                tools=[
                    vf.Tool(
                        name=tool.function.name,
                        description=tool.function.description,
                        parameters=tool.function.parameters,
                    )
                    for tool in tools
                ],
            )
            messages.extend(
                [
                    response.message,
                    vf.ToolMessage(
                        tool_call_id=response.message.tool_calls[0].id,
                        content="feedback",
                    ),
                ]
            )
    finally:
        await client.close()
    credit_routing.assign_advantages(trace, advantage)
    trained: list[int] = []
    for sample in native_trajectories.trace_to_samples(trace):
        assert sample.advantages is not None
        for token, mask, credit in zip(
            sample.token_ids, sample.mask, sample.advantages, strict=True
        ):
            if mask:
                assert credit == advantage
                trained.append(token)
    assert (
        trained == expected
    )  # Every completion, exactly once despite physical branches.
    assert all(not any(node.mask) for node in trace.nodes if not node.sampled)
    assert all(node.advantages is None for node in trace.nodes if not any(node.mask))


def test_prepare_rejects_unanswered_calls_before_rendering(
    experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = sft_config(experiment)
    assert config.data.dataset_from is not None
    path = config.data.dataset_from / "training.jsonl"
    record = load_json_lines(path, Conversation)[0]
    record = record.model_copy(
        update={
            "messages": [*record.messages[:3], record.messages[-1]],
            "requests": [
                record.requests[0],
                ConversationRequest(assistant_message_index=3, tools=[]),
            ],
        }
    )
    dataset.write_records(path, [record])
    monkeypatch.setattr(
        dataset, "load_student", Mock(side_effect=AssertionError("must not render"))
    )
    with pytest.raises(ValueError, match="outstanding tool calls"):
        run.run_experiment(experiment, RunRequest(RunStage.PREPARE))


def test_recorded_schema_references_are_not_fetched(
    tools: list[ToolDefinition], monkeypatch: pytest.MonkeyPatch
) -> None:
    record = conversation("task", "external-schema", tools)
    record.requests[0].tools[0] = tools[0].model_copy(
        update={
            "function": tools[0].function.model_copy(
                update={"parameters": {"$ref": "https://example.invalid/schema.json"}}
            )
        }
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", Mock(side_effect=AssertionError("must not fetch"))
    )
    with pytest.raises(ValueError, match="arguments or request schema are invalid"):
        dataset.validate_conversation(record)


@pytest.mark.parametrize("indices", [[], [2], [4, 2], [2, 2], [1, 4]])
def test_requests_cover_assistant_replies_in_order(
    tools: list[ToolDefinition], indices: list[int]
) -> None:
    record = conversation("task", "coverage", tools)
    record = record.model_copy(
        update={
            "requests": [
                ConversationRequest(assistant_message_index=index, tools=tools)
                for index in indices
            ]
        }
    )
    with pytest.raises(ValueError, match="cover each assistant message exactly once"):
        dataset.validate_conversation(record)
