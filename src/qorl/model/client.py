"""Typed model requests, local token budgets, and vLLM response normalization."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from qorl.exceptions import ContextBudgetError, ModelError, ModelRequestError
from qorl.model.schemas import (
    AdvertisedModel,
    GenerationRequest,
    GenerationResponse,
    JsonObject,
    LocalDecodingSettings,
    LocalServerIdentity,
    Message,
    MessageRole,
    ModelProvider,
    ModelSettings,
    TokenUsage,
    ToolCall,
)

JSON_OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
ERROR_DETAIL_LIMIT = 2_000


class ModelClient(Protocol):
    """Produce one assistant turn without executing any tools."""

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class ModelTransport(Protocol):
    """Exchange JSON with a configured model endpoint."""

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject: ...


class HttpTransport:
    """Send a single request; credentials are resolved from the environment in memory."""

    def __init__(self, settings: ModelSettings, *, api_key: str | None = None) -> None:
        if settings.base_url is None or settings.request_timeout_seconds is None:
            raise ValueError(
                "model API calls require model.base_url and model.request_timeout_seconds"
            )
        self.base_url = settings.base_url
        self.request_timeout_seconds = settings.request_timeout_seconds
        self.api_key_env = settings.api_key_env
        self._api_key = api_key

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
        """Validate JSON replies and distinguish fatal HTTP errors from transient ones."""
        api_key = self._api_key
        if api_key is None and self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ModelRequestError(f"missing model credential: {self.api_key_env}")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            urllib.parse.urljoin(self.base_url.rstrip("/") + "/", path),
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=headers,
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                return JSON_OBJECT.validate_json(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if api_key:
                detail = detail.replace(api_key, "[redacted]")
            error_type = (
                ModelRequestError
                if HTTPStatus.BAD_REQUEST
                <= error.code
                < HTTPStatus.INTERNAL_SERVER_ERROR
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
                else ModelError
            )
            raise error_type(
                f"model returned HTTP {error.code}: {detail[:ERROR_DETAIL_LIMIT]}"
            ) from None
        except (urllib.error.URLError, TimeoutError, ValidationError) as error:
            raise ModelError("model request failed or returned invalid JSON") from error


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
    tool_calls: list[ToolCall] | None = None


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    choices: list[ChatChoice] = Field(min_length=1, max_length=1)
    usage: ChatUsage | None = None


class LocalModelClient:
    """Use the served tokenizer and preserve reasoning in the next tool-call turn."""

    def __init__(
        self,
        model: ModelSettings,
        decoding: LocalDecodingSettings,
        *,
        transport: ModelTransport | None = None,
        served_model_name: str | None = None,
    ) -> None:
        if model.provider != ModelProvider.LOCAL:
            raise ValueError("local client requires model.provider=local")
        if model.base_url is None or model.request_timeout_seconds is None:
            raise ValueError(
                "model API calls require model.base_url and model.request_timeout_seconds"
            )
        if decoding.max_tokens > model.context_length:
            raise ValueError("decoding.max_tokens exceeds model.context_length")
        self.model = model
        self.decoding = decoding
        self.base_url = model.base_url
        self.served_model_name = served_model_name or model.name_or_path
        self.transport = transport or HttpTransport(model)
        self.identity: LocalServerIdentity | None = None

    def preflight(self) -> LocalServerIdentity:
        """Check advertised model/context and record, rather than pin, runtime vLLM."""
        try:
            models = ModelList.model_validate(self.transport.request("models")).data
        except ValidationError as error:
            raise ModelError("model server returned an invalid model list") from error
        selected = next(
            (model for model in models if model.id == self.served_model_name), None
        )
        if selected is None:
            raise ModelError(
                f"model server does not advertise {self.served_model_name}"
            )
        context_model = selected
        if selected.parent:
            parent = next(
                (model for model in models if model.id == selected.parent), None
            )
            if parent is None:
                raise ModelError(
                    f"model server does not advertise adapter parent {selected.parent}"
                )
            context_model = parent
        if (
            selected.max_model_len is not None
            and selected.max_model_len != self.model.context_length
        ):
            raise ModelError(
                "selected model context length differs from model.context_length"
            )
        if context_model.max_model_len != self.model.context_length:
            raise ModelError(
                f"model context length mismatch: expected={self.model.context_length} "
                f"actual={context_model.max_model_len}"
            )
        try:
            version = VersionResponse.model_validate(
                self.transport.request("../version")
            )
        except ValidationError as error:
            raise ModelError(
                "model server returned an invalid version response"
            ) from error
        self.identity = LocalServerIdentity(
            base_url=self.base_url,
            model=selected,
            context_model=context_model,
            vllm_version=version.version,
        )
        return self.identity

    def request_body(self, request: GenerationRequest) -> JsonObject:
        """Render the local API format without changing tool schemas or argument strings."""
        messages: list[JsonValue] = []
        for message in request.messages:
            item = JSON_OBJECT.validate_python(
                message.model_dump(mode="json", exclude_none=True)
            )
            if message.content is None:
                item["content"] = None
            if message.reasoning_content is not None:
                # vLLM 0.28 copies reasoning to reasoning_content for HF templates.
                item["reasoning"] = item.pop("reasoning_content")
            messages.append(item)
        body: JsonObject = {
            "model": self.served_model_name,
            "messages": messages,
            "tools": [
                JSON_OBJECT.validate_python(tool.model_dump(mode="json"))
                for tool in request.tools
            ],
            "tool_choice": "required" if request.tools else "none",
            "parallel_tool_calls": False,
            **JSON_OBJECT.validate_python(
                self.decoding.model_dump(exclude={"thinking"})
            ),
            "chat_template_kwargs": {"enable_thinking": self.decoding.thinking},
        }
        if request.seed is not None:
            body["seed"] = request.seed
        return body

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Count the complete rendered prompt before generation; never trim history."""
        body = self.request_body(request)
        token_request = {
            name: body[name]
            for name in ("model", "messages", "tools", "chat_template_kwargs")
        }
        try:
            counted = TokenCount.model_validate(
                self.transport.request("../tokenize", token_request)
            )
        except ValidationError as error:
            raise ModelError("model server returned an invalid token count") from error
        if counted.max_model_len != self.model.context_length:
            raise ModelError(
                "tokenizer context length differs from model.context_length"
            )
        if counted.count + self.decoding.max_tokens > self.model.context_length:
            raise ContextBudgetError(
                counted.count, self.decoding.max_tokens, self.model.context_length
            )
        raw = self.transport.request("chat/completions", body)
        try:
            response = ChatResponse.model_validate(raw)
        except ValidationError as error:
            raise ModelError("model returned an invalid chat completion") from error
        choice = response.choices[0]
        if choice.message.role != MessageRole.ASSISTANT:
            raise ModelError("model returned a non-assistant message")
        usage = response.usage or ChatUsage()
        return GenerationResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=choice.message.content,
                tool_calls=choice.message.tool_calls,
                reasoning_content=choice.message.reasoning,
            ),
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else None,
                cached_tokens=usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                else None,
            ),
            finish_reason=choice.finish_reason,
            truncated=choice.finish_reason == "length",
            prompt_tokens=counted.count,
            requested_max_tokens=self.decoding.max_tokens,
            request=body,
            raw_response=raw,
        )
