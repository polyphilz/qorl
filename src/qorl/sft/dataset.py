"""Generate or reuse frozen conversations, then render and pack them for a student."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import distribution, version
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from typing import Protocol

import renderers.base as rendering
from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import SchemaError, ValidationError
from prime_rl.trainer.sft.data import CatDataset, Sample, StatefulIterableDataset
from pydantic import BaseModel, JsonValue, TypeAdapter
from referencing import Registry
from referencing.exceptions import Unresolvable
from renderers.base import (
    Message as RendererMessage,
)
from renderers.base import (
    Renderer,
    ToolCallFunction,
    ToolSpec,
    build_training_sample,
)
from renderers.base import (
    ToolCall as RendererToolCall,
)
from renderers.configs import AutoRendererConfig, RendererConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from qorl.experiment.schemas import SftExperimentConfig
from qorl.model.files import resolve_model
from qorl.model.schemas import LocalInferenceSettings, MessageRole, ToolDefinition
from qorl.paths import REPOSITORY_ROOT
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    Conversation,
    ConversationRequest,
    DatasetPreparationIdentity,
    DatasetPreparationReport,
    ImportedGenerationSeeds,
    PackingRowMetadata,
    PreparedDatasetManifest,
    PreparedSplitReport,
    RenderedConversation,
    RenderedRequest,
    RenderingRejection,
    TrainingRow,
)
from qorl.taskset.schemas import TaskRole, TaskSelection
from qorl.util.hashing import sha256_bytes, sha256_file, sha256_json
from qorl.util.io import write_json

RENDERER_CONFIG: TypeAdapter[RendererConfig] = TypeAdapter(RendererConfig)
SPLIT_NAMES = {TaskRole.TRAIN: "training", TaskRole.VALIDATION: "validation"}
RECORD_NUMBER_WIDTH = 6


class TokenizerLoader(Protocol):
    """The offline Hugging Face loader boundary, before checking tokenizer type."""

    def from_pretrained(
        self, path: str, /, *, local_files_only: bool, use_fast: bool
    ) -> object: ...


class TokenizerBackend(Protocol):
    """The tokenizer engine's serialized state used in preparation identity."""

    def to_str(self) -> str: ...


class StudentTokenizer(Protocol):
    """The fast tokenizer operations used for text preparation and inspection."""

    @property
    def backend_tokenizer(self) -> TokenizerBackend: ...

    @property
    def all_special_tokens(self) -> list[str]: ...

    @property
    def chat_template(self) -> str | dict[str, str] | None: ...

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str | list[str]: ...


class RendererFactory(Protocol):
    """The native renderer factory's supported text-tokenizer inputs."""

    def create_renderer(
        self, tokenizer: PreTrainedTokenizerFast, config: RendererConfig
    ) -> Renderer: ...


TOKENIZER_LOADER: TokenizerLoader = AutoTokenizer
RENDERER_FACTORY: RendererFactory = rendering


@dataclass(frozen=True)
class ConversationSource:
    """Validated source files and their conversations, held without changing them."""

    directory: Path
    manifest: PreparedDatasetManifest
    conversations: dict[TaskRole, list[Conversation]]
    files: dict[str, bytes]


@dataclass(frozen=True)
class StudentRenderer:
    """The student's tokenizer, native renderer, and resolved rendering settings."""

    tokenizer: StudentTokenizer
    renderer: Renderer
    config: RendererConfig


def source_file(directory: Path, relative: Path) -> Path:
    """Require a source artifact's files to exist inside that artifact."""
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(
            f"dataset file must be relative and inside its artifact: {relative}"
        )
    if not path.is_file():
        raise ValueError(f"dataset file is missing: {path}")
    return path


def load_source(
    config: SftExperimentConfig,
    selections: dict[TaskRole, TaskSelection],
    *,
    generated_source: Path | None = None,
) -> ConversationSource:
    """Check frozen split membership, original seeds and tool references, not outcomes."""
    source = config.data.dataset_from or generated_source
    if source is None:
        raise ValueError(
            "generate a conversation artifact before loading the student source"
        )
    directory = (REPOSITORY_ROOT / source.expanduser()).resolve()
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = PreparedDatasetManifest.model_validate_json(manifest_bytes)
    files = {"source-manifest.json": manifest_bytes}
    conversations: dict[TaskRole, list[Conversation]] = {}
    seen: set[str] = set()
    seeds = (
        config.data.imported_generation_seeds
        if config.data.dataset_from is not None
        else ImportedGenerationSeeds(
            training=config.experiment.seed, validation=config.experiment.seed
        )
    )
    if seeds is None:
        raise ValueError("reused conversations require original generation seeds")
    for role, split, configured, generation_seed in (
        (TaskRole.TRAIN, manifest.training, config.data.training, seeds.training),
        (
            TaskRole.VALIDATION,
            manifest.validation,
            config.data.validation,
            seeds.validation,
        ),
    ):
        if (
            split.selection != selections[role]
            or split.selection_seed != configured.selection_seed
            or split.generation_seed != generation_seed
            or split.selection_expression != configured.expression
        ):
            raise ValueError(
                f"source {role.value} selection or seeds changed; create a new experiment"
            )
        path = source_file(directory, split.conversations)
        content = path.read_bytes()
        files[f"{SPLIT_NAMES[role]}.jsonl"] = content
        records = [
            Conversation.model_validate_json(line)
            for line in content.splitlines()
            if line.strip()
        ]
        for record in records:
            if record.conversation_id in seen:
                raise ValueError(f"repeated conversation_id: {record.conversation_id}")
            seen.add(record.conversation_id)
            if record.task_id not in split.selection.task_ids:
                raise ValueError(
                    f"{record.conversation_id}: task {record.task_id} is outside its {role.value} selection"
                )
            validate_conversation(record)
        conversations[role] = records
    return ConversationSource(directory, manifest, conversations, files)


def validate_conversation(record: Conversation) -> None:
    """Require request-specific tools and answered calls before advancing history."""
    assistant_indices = [
        index
        for index, message in enumerate(record.messages)
        if message.role == MessageRole.ASSISTANT
    ]
    if [
        request.assistant_message_index for request in record.requests
    ] != assistant_indices:
        raise ValueError(
            f"{record.conversation_id}: requests must cover each assistant message exactly once, in order"
        )
    available: dict[int, dict[str, ToolDefinition]] = {}
    for request in record.requests:
        names = {tool.function.name: tool for tool in request.tools}
        if len(names) != len(request.tools):
            raise ValueError(
                f"{record.conversation_id}: request tool definitions contain repeated names"
            )
        available[request.assistant_message_index] = names
    pending: set[str] = set()
    calls: set[str] = set()
    for index, message in enumerate(record.messages):
        if pending and message.role != MessageRole.TOOL:
            raise ValueError(
                f"{record.conversation_id}: outstanding tool calls require results before message {index}"
            )
        if message.tool_calls and message.role != MessageRole.ASSISTANT:
            raise ValueError(
                f"{record.conversation_id}: only assistant messages can call tools"
            )
        for call in message.tool_calls or []:
            if call.id in calls or call.function.name not in available[index]:
                raise ValueError(
                    f"{record.conversation_id}: repeated call ID or unavailable tool {call.function.name} at message {index}"
                )
            arguments = JSON_OBJECT_ADAPTER.validate_json(call.function.arguments)
            schema = available[index][call.function.name].function.parameters
            try:
                validate(
                    arguments,
                    schema,
                    cls=Draft202012Validator,
                    registry=Registry[JsonValue](),
                )
            except (SchemaError, ValidationError, Unresolvable) as error:
                raise ValueError(
                    f"{record.conversation_id}: {call.function.name} arguments or request schema are invalid: {error}"
                ) from error
            calls.add(call.id)
            pending.add(call.id)
        if message.role == MessageRole.TOOL:
            if message.tool_call_id not in pending:
                raise ValueError(
                    f"{record.conversation_id}: tool result has no unanswered call"
                )
            pending.remove(message.tool_call_id)


def load_student(config: SftExperimentConfig) -> StudentRenderer:
    """Load a cached tokenizer and native renderer; do not load model tensors."""
    base = resolve_model(config.model)
    tokenizer = TOKENIZER_LOADER.from_pretrained(
        str(base), local_files_only=True, use_fast=True
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("student preparation requires a fast text tokenizer")
    if not isinstance(config.inference, LocalInferenceSettings):
        raise ValueError("SFT preparation requires a local student")
    selected = config.training.renderer
    if isinstance(selected, AutoRendererConfig):
        name = rendering.MODEL_RENDERER_MAP.get(config.model.name_or_path)
        if name is None:
            raise ValueError(
                "unknown student renderer; set training.renderer explicitly for this model"
            )
        selected = RENDERER_CONFIG.validate_python(
            {**selected.model_dump(exclude_unset=True), "name": name}
        )
    options = JSON_OBJECT_ADAPTER.validate_python(
        selected.model_dump(mode="json", exclude_unset=True)
    )
    options["name"] = selected.name
    if "enable_thinking" in selected.template_field_names():
        options["enable_thinking"] = config.inference.thinking
    elif config.inference.thinking:
        raise ValueError(
            "student renderer does not support the configured thinking switch"
        )
    resolved = RENDERER_CONFIG.validate_python(options)
    renderer = RENDERER_FACTORY.create_renderer(tokenizer, resolved)
    return StudentRenderer(tokenizer, renderer, resolved)


def rendering_messages(record: Conversation) -> list[RendererMessage]:
    """Translate normalized text/tool messages into the renderer's native input."""
    result: list[RendererMessage] = []
    for message in record.messages:
        item = RendererMessage(role=message.role.value, content=message.content or "")
        if message.reasoning_content is not None:
            item["reasoning_content"] = message.reasoning_content
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            item["name"] = message.name
        if message.tool_calls:
            item["tool_calls"] = [
                RendererToolCall(
                    id=call.id,
                    type=call.type,
                    function=ToolCallFunction(
                        name=call.function.name,
                        arguments=JSON_OBJECT_ADAPTER.validate_json(
                            call.function.arguments
                        ),
                    ),
                )
                for call in message.tool_calls
            ]
        result.append(item)
    return result


def render_request(
    record: Conversation,
    request: ConversationRequest,
    student: StudentRenderer,
) -> RenderedRequest:
    """Render one reply against its actual tools; prior assistant turns are context."""
    messages = rendering_messages(record)[: request.assistant_message_index + 1]
    definitions = [ToolSpec(**tool.model_dump(mode="json")) for tool in request.tools]
    rendered = student.renderer.render(messages, tools=definitions)
    sample = build_training_sample(
        student.renderer,
        messages,
        tools=definitions,
        ensure_final_stop=True,
        role_to_mask=lambda message: message is messages[-1],
    )
    indices = list(rendered.message_indices)
    if len(sample.token_ids) == len(rendered.token_ids) + 1:
        indices.append(len(messages) - 1)
    if (
        len(indices) != len(sample.token_ids)
        or sample.token_ids[: len(rendered.token_ids)] != rendered.token_ids
    ):
        raise ValueError(
            f"{record.conversation_id}: renderer returned inconsistent tokens or attribution"
        )
    if sample.multi_modal_data is not None or len(sample.token_ids) <= 1:
        raise ValueError(
            f"{record.conversation_id}: expected a nonempty text-only training conversation"
        )
    row = TrainingRow(
        input_ids=sample.token_ids[:-1],
        target_ids=sample.token_ids[1:],
        loss_mask=sample.loss_mask[1:],
        position_ids=list(range(len(sample.token_ids) - 1)),
        seq_lens=[len(sample.token_ids) - 1],
    )
    return RenderedRequest(
        conversation_id=record.conversation_id,
        assistant_message_index=request.assistant_message_index,
        tools_sha256=sha256_json(
            [tool.model_dump(mode="json") for tool in request.tools]
        ),
        rendered_text=TypeAdapter(str).validate_python(
            student.tokenizer.decode(sample.token_ids, skip_special_tokens=False)
        ),
        target_message_indices=indices[1:],
        sample=row,
    )


def render_conversation(
    record: Conversation, student: StudentRenderer, context_length: int
) -> RenderedConversation:
    """Keep all request samples together when rejecting an overlong conversation."""
    requests = [render_request(record, request, student) for request in record.requests]
    rejection = None
    if any(len(request.sample.input_ids) > context_length for request in requests):
        rejection = RenderingRejection.OVER_CONTEXT
    elif not requests or any(not any(request.sample.loss_mask) for request in requests):
        rejection = RenderingRejection.NO_TARGETS
    return RenderedConversation(
        conversation_id=record.conversation_id,
        task_id=record.task_id,
        requests=requests,
        rejection=rejection,
    )


class RenderedDataset(StatefulIterableDataset):
    """Finite pre-rendered samples for Prime-RL's unchanged CatDataset packer."""

    def __init__(self, records: list[RenderedRequest]):
        super().__init__()
        self.records = records

    def __iter__(self) -> Iterator[Sample]:
        """Yield independent, already-shifted requests in the supplied order."""
        for record in self.records:
            yield Sample(**record.sample.model_dump())


def training_rows(samples: Iterable[object]) -> Iterator[TrainingRow]:
    """Validate native trainer records before using their fields in preparation."""
    for sample in samples:
        yield TrainingRow.model_validate(sample)


def pack_conversations(
    records: list[RenderedConversation], context_length: int, seed: int
) -> tuple[list[TrainingRow], list[PackingRowMetadata]]:
    """Pack requests from accepted conversations, retaining unpadded boundaries."""
    ordered = [
        request
        for record in records
        if record.rejection is None
        for request in record.requests
    ]
    if any(len(record.sample.input_ids) > context_length for record in ordered):
        raise ValueError("overlong conversations must be rejected before packing")
    Random(seed).shuffle(ordered)
    rows: list[TrainingRow] = []
    metadata: list[PackingRowMetadata] = []
    offset = 0
    for row in training_rows(
        CatDataset(RenderedDataset(ordered), seq_len=context_length)
    ):
        members = ordered[offset : offset + len(row.seq_lens)]
        lengths = [len(record.sample.input_ids) for record in members]
        rows.append(row)
        metadata.append(
            PackingRowMetadata(
                conversation_ids=[record.conversation_id for record in members],
                assistant_message_indices=[
                    record.assistant_message_index for record in members
                ],
                request_lengths=lengths,
                padding_tokens=context_length - sum(lengths),
            )
        )
        offset += len(members)
    if offset != len(ordered):
        raise ValueError("native packer did not account for all accepted requests")
    return rows, metadata


def preparation_identity(
    config: SftExperimentConfig, source: ConversationSource, student: StudentRenderer
) -> DatasetPreparationIdentity:
    """Bind resumable work to source bytes, settings, tokenizer and installed code."""
    tokenizer = student.tokenizer
    tokenizer_data = JSON_OBJECT_ADAPTER.validate_python(
        {
            "backend": tokenizer.backend_tokenizer.to_str(),
            "special_tokens": tokenizer.all_special_tokens,
            "chat_template": tokenizer.chat_template,
        }
    )
    dependencies = {
        name: version(name)
        for name in (
            "prime-rl",
            "renderers",
            "transformers",
            "tokenizers",
            "torchdata",
            "jsonschema",
            "referencing",
        )
    }
    for name in ("prime-rl", "renderers"):
        dependencies[f"{name}-source"] = (
            distribution(name).read_text("direct_url.json") or ""
        )
    return DatasetPreparationIdentity(
        config_sha256=sha256_json(config.model_dump(mode="json")),
        source=source.directory,
        source_files={
            name: sha256_bytes(content) for name, content in source.files.items()
        },
        tokenizer_sha256=sha256_json(tokenizer_data),
        renderer=JSON_OBJECT_ADAPTER.validate_python(
            student.config.model_dump(mode="json")
        ),
        dependencies=dependencies,
    )


def write_bytes(path: Path, content: bytes) -> None:
    """Publish an entire preparation artifact atomically, including copied sources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
        pending = Path(temporary.name)
    pending.replace(path)


def write_records(path: Path, records: Sequence[BaseModel]) -> None:
    """Publish JSONL only after all rows have been serialized."""
    write_bytes(
        path, "".join(record.model_dump_json() + "\n" for record in records).encode()
    )


def prepare_dataset(
    config: SftExperimentConfig,
    selections: dict[TaskRole, TaskSelection],
    output: Path,
    *,
    resume: bool = False,
) -> DatasetPreparationReport:
    """Reuse immutable sources and completed renders; publish a report after packing."""
    if output.exists() and not resume:
        raise ValueError(
            "dataset preparation already exists; use --resume or start a new run"
        )
    generated = None
    if config.data.dataset_from is None:
        from qorl.sft.generate import generate_dataset

        generated = generate_dataset(config, selections, output.parent / "generation")
    source = load_source(config, selections, generated_source=generated)
    student = load_student(config)
    identity = preparation_identity(config, source, student)
    identity_path = output / "preparation.json"
    if output.exists():
        saved = DatasetPreparationIdentity.model_validate_json(
            identity_path.read_bytes()
        )
        if saved != identity:
            raise ValueError("dataset preparation inputs changed; start a new run")
    else:
        output.mkdir(parents=True)
        write_json(identity_path, identity.model_dump(mode="json"))
    report_path = output / "report.json"
    if report_path.exists():
        report = DatasetPreparationReport.model_validate_json(report_path.read_bytes())
        for name, checksum in report.files.items():
            if sha256_file(source_file(output, Path(name))) != checksum:
                raise ValueError(f"prepared dataset file changed: {name}")
        require_both_splits(report)
        return report
    for name, content in source.files.items():
        write_bytes(output / name, content)
    manifest = source.manifest.model_copy(
        update={
            "training": source.manifest.training.model_copy(
                update={"conversations": Path("training.jsonl")}
            ),
            "validation": source.manifest.validation.model_copy(
                update={"conversations": Path("validation.jsonl")}
            ),
        }
    )
    write_json(output / "manifest.json", manifest.model_dump(mode="json"))
    reports: dict[TaskRole, PreparedSplitReport] = {}
    for role, conversations in source.conversations.items():
        directory = output / SPLIT_NAMES[role]
        records: list[RenderedConversation] = []
        for index, conversation in enumerate(conversations):
            path = directory / "rendered" / f"{index:0{RECORD_NUMBER_WIDTH}d}.json"
            if path.exists():
                rendered = RenderedConversation.model_validate_json(path.read_bytes())
                if (rendered.conversation_id, rendered.task_id) != (
                    conversation.conversation_id,
                    conversation.task_id,
                ):
                    raise ValueError(
                        f"saved rendering belongs to a different conversation: {path}"
                    )
            else:
                rendered = render_conversation(
                    conversation, student, config.model.context_length
                )
                write_json(path, rendered.model_dump(mode="json"))
            records.append(rendered)
        rows, metadata = pack_conversations(
            records, config.model.context_length, config.experiment.seed
        )
        write_records(directory / "packed.jsonl", list(rows))
        write_records(directory / "packing.jsonl", list(metadata))
        accepted = [record for record in records if record.rejection is None]
        reports[role] = PreparedSplitReport(
            selected_tasks=len(selections[role].task_ids),
            source_conversations=len(records),
            accepted_conversations=len(accepted),
            accepted_requests=sum(len(record.requests) for record in accepted),
            accepted_tasks=len({record.task_id for record in accepted}),
            rejections={
                reason: sum(record.rejection == reason for record in records)
                for reason in RenderingRejection
            },
            packed_rows=len(rows),
            input_tokens=sum(
                len(request.sample.input_ids)
                for record in accepted
                for request in record.requests
            ),
            supervised_tokens=sum(sum(row.loss_mask) for row in rows),
            padding_tokens=sum(item.padding_tokens for item in metadata),
        )
    report = DatasetPreparationReport(
        context_length=config.model.context_length,
        shuffle_seed=config.experiment.seed,
        training=reports[TaskRole.TRAIN],
        validation=reports[TaskRole.VALIDATION],
        files={
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        },
    )
    write_json(report_path, report.model_dump(mode="json"))
    require_both_splits(report)
    return report


def require_both_splits(report: DatasetPreparationReport) -> None:
    """Keep an empty-split report inspectable, but do not call it trainable data."""
    if not report.training.packed_rows or not report.validation.packed_rows:
        raise ValueError(
            "preparation produced an empty training or validation split; see dataset/report.json"
        )
