"""Model identities, API access, and inference settings owned by an experiment."""

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from qorl.inference.schemas import ServingSettings

type JsonObject = dict[str, JsonValue]


class ModelProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


OPENROUTER_MODEL_ID = "qwen/qwen3.8-2.4t-a95b"
OPENROUTER_CONTEXT_LIMIT = 1_000_000
OPENROUTER_OUTPUT_LIMIT = 131_072


class RetrySettings(BaseModel):
    """Bound transport retries, not content generations."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_attempts: int = Field(default=3, ge=1)
    initial_delay_seconds: float = Field(default=2.0, gt=0)
    maximum_delay_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def bounded_delay(self) -> Self:
        if self.initial_delay_seconds > self.maximum_delay_seconds:
            raise ValueError("initial retry delay exceeds maximum retry delay")
        return self


class ModelSettings(BaseModel):
    """A model identity, with API access fields required when calling the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ModelProvider
    name_or_path: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    revision: str | None = None
    adapter_path: Path | None = None
    base_url: str | None = None
    request_timeout_seconds: int | None = Field(default=None, gt=0)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    max_concurrent_requests: int = Field(default=8, ge=1)
    retry: RetrySettings = RetrySettings()

    @model_validator(mode="after")
    def plain_http_address(self) -> Self:
        """Keep credentials and query parameters out of saved connection URLs."""
        if self.base_url is None:
            return self
        address = urlsplit(self.base_url)
        _ = (
            address.port
        )  # Reject malformed or out-of-range ports during config loading.
        if (
            address.scheme not in ("http", "https")
            or not address.hostname
            or address.username is not None
            or address.password is not None
            or address.query
            or address.fragment
        ):
            raise ValueError(
                "model.base_url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        return self


class LocalInferenceSettings(BaseModel):
    """Generation settings sent to the local server, plus that server's own settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0, le=1)
    presence_penalty: float
    repetition_penalty: float = Field(gt=0)
    thinking: bool
    serving: ServingSettings


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class AstraInferenceSettings(BaseModel):
    """Astra effort and optional provider-generated summaries, not full reasoning.

    Summaries cannot recover missing reasoning text in previously saved traces.
    Student thinking settings are independent of these provider options.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(gt=0)
    reasoning_effort: ReasoningEffort
    reasoning_summary: Literal["auto"] | None = None


class OpenRouterInferenceSettings(BaseModel):
    """Qwen requires reasoning; these are the supported effort levels."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_tokens: int = Field(gt=0, le=OPENROUTER_OUTPUT_LIMIT)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    reasoning_effort: Literal[
        ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.XHIGH
    ]

    def validate_model(self, model: ModelSettings) -> None:
        """Apply the same hosted identity/capacity contract in configs and clients."""
        if (
            model.provider != ModelProvider.OPENROUTER
            or model.name_or_path != OPENROUTER_MODEL_ID
        ):
            raise ValueError(f"OpenRouter supports only {OPENROUTER_MODEL_ID}")
        if model.adapter_path is not None or model.revision is not None:
            raise ValueError("hosted models do not accept revisions or adapters")
        if (
            not model.base_url
            or not model.request_timeout_seconds
            or not model.api_key_env
        ):
            raise ValueError(
                "hosted models require base_url, request_timeout_seconds, and api_key_env"
            )
        if model.context_length > OPENROUTER_CONTEXT_LIMIT:
            raise ValueError("model.context_length exceeds OpenRouter model capacity")
        if self.max_tokens > model.context_length:
            raise ValueError("inference.max_tokens exceeds model.context_length")


type InferenceSettings = (
    LocalInferenceSettings | AstraInferenceSettings | OpenRouterInferenceSettings
)


class ModelPreset(BaseModel):
    """Connection and inference values copied into an experiment at creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelSettings
    inference: InferenceSettings


class FunctionCall(BaseModel):
    """A model's function name and original, unmodified JSON arguments string."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ResponsesContinuation(BaseModel):
    """Provider-owned output items, including encrypted reasoning and message phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    output: list[JsonObject]


class OpenRouterContinuation(BaseModel):
    """Provider-owned reasoning blocks; replay without converting or stripping keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    reasoning_details: list[JsonObject] | None


class Message(BaseModel):
    """Text/tool conversation shared with the agent and local chat template."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    continuation: ResponsesContinuation | OpenRouterContinuation | None = None


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: JsonObject


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["function"] = "function"
    function: ToolFunction


class GenerationRequest(BaseModel):
    """One model turn; the agent supplies messages and the tools currently allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[Message]
    tools: list[ToolDefinition]
    seed: int | None = None


class TokenUsage(BaseModel):
    """Normalized usage; missing provider counts remain unknown, not zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class GenerationResponse(BaseModel):
    """Normalized assistant output plus the complete request/response evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Message
    usage: TokenUsage
    finish_reason: str
    truncated: bool
    prompt_tokens: int = Field(ge=0)
    requested_max_tokens: int = Field(gt=0)
    request: JsonObject
    raw_response: JsonObject


class ModelResponseFailure(BaseModel):
    """A received reply that cannot be used, retained before propagating failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: JsonObject
    raw_response: JsonObject
    error: str
    usage: TokenUsage = TokenUsage()


class AdvertisedModel(BaseModel):
    """Identity fields returned by vLLM's model-list endpoint."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None


class LocalServerIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    model: AdvertisedModel
    context_model: AdvertisedModel
    vllm_version: str


class ModelWeightIndex(BaseModel):
    """Shard filenames from a Hugging Face model's weights index."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    weight_map: dict[str, str] = Field(min_length=1)


class TokenCount(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    count: int = Field(ge=0)
    max_model_len: int = Field(gt=0)


class ModelList(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: list[AdvertisedModel]


class VersionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    version: str = Field(min_length=1)


class CompletionDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    reasoning_tokens: int | None = Field(default=None, ge=0)


class PromptDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    cached_tokens: int | None = Field(default=None, ge=0)


class ChatUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    completion_tokens_details: CompletionDetails | None = None
    prompt_tokens_details: PromptDetails | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    role: str
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    choices: list[ChatChoice] = Field(min_length=1, max_length=1)
    usage: ChatUsage | None = None


class OpenRouterMessage(ChatMessage):
    reasoning_details: list[JsonObject] | None = None


class OpenRouterChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: OpenRouterMessage
    finish_reason: str
    error: JsonObject | None = None


class OpenRouterReply(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str = Field(min_length=1)
    provider: str | None = None
    choices: list[OpenRouterChoice] = Field(min_length=1, max_length=1)
    usage: ChatUsage | None = None
    error: JsonObject | None = None


class InputTokenCount(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    input_tokens: int = Field(ge=0)


class ResponsesUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_tokens_details: PromptDetails | None = None
    output_tokens_details: CompletionDetails | None = None


class ResponseItem(BaseModel):
    """Fields inspected by QORL; complete provider items are retained separately."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str
    role: str | None = None
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None
    content: list[JsonObject] = Field(default_factory=list[JsonObject])
    summary: list[JsonObject] = Field(default_factory=list[JsonObject])


class ResponsesReply(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    status: str
    output: list[JsonObject]
    usage: ResponsesUsage | None = None
    incomplete_details: JsonObject | None = None
