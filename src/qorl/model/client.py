"""Local and Astra tool calling, exact input budgets, and bounded HTTP retries."""

import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from threading import BoundedSemaphore
from typing import Protocol
from uuid import uuid4

from pydantic import (
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from qorl.model.exceptions import (
    ContextBudgetError,
    ModelError,
    ModelRequestError,
    TransientModelError,
)
from qorl.model.schemas import (
    AstraInferenceSettings,
    ChatResponse,
    ChatUsage,
    FunctionCall,
    GenerationRequest,
    GenerationResponse,
    InferenceSettings,
    InputTokenCount,
    JsonObject,
    LocalInferenceSettings,
    LocalServerIdentity,
    Message,
    MessageRole,
    ModelList,
    ModelProvider,
    ModelSettings,
    ResponseItem,
    ResponsesContinuation,
    ResponsesReply,
    TokenCount,
    TokenUsage,
    ToolCall,
    VersionResponse,
)

JSON_OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
ERROR_DETAIL_LIMIT = 2_000
RETRY_BACKOFF_FACTOR = 2
ASTRA_MODEL_ID = "gpt-6-astra"
ASTRA_CONTEXT_LIMIT = 1_050_000
ASTRA_INPUT_LIMIT = 922_000
ASTRA_OUTPUT_LIMIT = 128_000


class ModelClient(Protocol):
    """Produce one assistant turn without executing any tools."""

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class ModelTransport(Protocol):
    """Exchange JSON with a configured model endpoint."""

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject: ...


class HttpTransport:
    """Bound concurrent requests and retry transient failures without regenerating content."""

    def __init__(
        self,
        settings: ModelSettings,
        *,
        api_key: str | None = None,
        semaphore: BoundedSemaphore | None = None,
    ) -> None:
        if settings.base_url is None or settings.request_timeout_seconds is None:
            raise ValueError(
                "model API calls require model.base_url and model.request_timeout_seconds"
            )
        self.base_url = settings.base_url
        self.request_timeout_seconds = settings.request_timeout_seconds
        self.api_key_env = settings.api_key_env
        self._api_key = api_key
        self.retry = settings.retry
        self._semaphore = semaphore or BoundedSemaphore(
            settings.max_concurrent_requests
        )

    def request(self, path: str, body: JsonObject | None = None) -> JsonObject:
        """Retry only transport failures; never retry a completed model response."""
        # The training proxy replays retries instead of recording unseen completions.
        idempotency_key = uuid4().hex
        delay = self.retry.initial_delay_seconds
        for attempt in range(self.retry.max_attempts):
            try:
                with self._semaphore:
                    return self._request(path, body, idempotency_key)
            except TransientModelError as error:
                if attempt + 1 == self.retry.max_attempts:
                    raise
                wait = random.uniform(0, delay)
                if error.retry_after_seconds is not None:
                    if error.retry_after_seconds > self.retry.maximum_delay_seconds:
                        raise
                    wait = max(wait, error.retry_after_seconds)
                time.sleep(wait)
                delay = min(
                    delay * RETRY_BACKOFF_FACTOR, self.retry.maximum_delay_seconds
                )
        raise AssertionError("retry settings must allow at least one attempt")

    def _request(
        self, path: str, body: JsonObject | None, idempotency_key: str
    ) -> JsonObject:
        """Send one request and validate its JSON; credentials stay in memory."""
        api_key = self._api_key
        if api_key is None and self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ModelRequestError(f"missing model credential: {self.api_key_env}")
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
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
            message = f"model returned HTTP {error.code}: {detail[:ERROR_DETAIL_LIMIT]}"
            if (
                error.code == HTTPStatus.TOO_MANY_REQUESTS
                or error.code >= HTTPStatus.INTERNAL_SERVER_ERROR
            ):
                raise TransientModelError(
                    message,
                    retry_after_seconds(
                        error.headers.get("Retry-After") if error.headers else None
                    ),
                ) from None
            raise ModelRequestError(message) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TransientModelError("model transport failed") from error
        except ValidationError as error:
            raise ModelError("model returned invalid JSON") from error


def retry_after_seconds(value: str | None) -> float | None:
    """Read either form of Retry-After; ignore malformed headers."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


class LocalModelClient:
    """Use the served tokenizer and preserve reasoning in the next tool-call turn."""

    def __init__(
        self,
        model: ModelSettings,
        inference: LocalInferenceSettings,
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
        if inference.max_tokens > model.context_length:
            raise ValueError("inference.max_tokens exceeds model.context_length")
        self.model = model
        self.inference = inference
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
            if message.continuation is not None:
                raise ModelError(
                    "Responses continuation cannot be sent to a local model"
                )
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
            "tool_choice": "auto" if request.tools else "none",
            "parallel_tool_calls": True,
            # Server settings configure vLLM's process; only sampling fields are sent.
            **JSON_OBJECT.validate_python(
                self.inference.model_dump(exclude={"thinking", "serving"})
            ),
            "chat_template_kwargs": {"enable_thinking": self.inference.thinking},
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
        if counted.count + self.inference.max_tokens > self.model.context_length:
            raise ContextBudgetError(
                counted.count, self.inference.max_tokens, self.model.context_length
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
                reasoning_content=choice.message.reasoning
                or choice.message.reasoning_content,
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
            requested_max_tokens=self.inference.max_tokens,
            request=body,
            raw_response=raw,
        )


class AstraModelClient:
    """Stateless Responses tool calling with complete, encrypted continuation history."""

    def __init__(
        self,
        model: ModelSettings,
        inference: AstraInferenceSettings,
        *,
        transport: ModelTransport | None = None,
    ) -> None:
        if (
            model.provider != ModelProvider.OPENAI
            or model.name_or_path != ASTRA_MODEL_ID
        ):
            raise ValueError("OpenAI tool calling supports only gpt-6-astra")
        if model.revision is not None or model.adapter_path is not None:
            raise ValueError("hosted models do not accept revisions or adapters")
        if (
            not model.base_url
            or not model.request_timeout_seconds
            or not model.api_key_env
        ):
            raise ValueError(
                "hosted models require base_url, request_timeout_seconds, and api_key_env"
            )
        if model.context_length > ASTRA_CONTEXT_LIMIT:
            raise ValueError("model.context_length exceeds Astra's context capacity")
        if inference.max_tokens > min(model.context_length, ASTRA_OUTPUT_LIMIT):
            raise ValueError(
                "inference.max_tokens exceeds the context or output capacity"
            )
        self.model = model
        self.inference = inference
        self.transport = transport or HttpTransport(model)

    def request_body(self, request: GenerationRequest) -> JsonObject:
        """Replay provider output items unchanged; do not rebuild reasoning or tool calls."""
        inputs: list[JsonValue] = []
        for message in request.messages:
            if message.continuation is not None:
                if (
                    message.role != MessageRole.ASSISTANT
                    or message.continuation.model != self.model.name_or_path
                ):
                    raise ModelError(
                        "Responses continuation belongs to a different model or role"
                    )
                inputs.extend(message.continuation.output)
            elif message.role == MessageRole.TOOL:
                if message.tool_call_id is None or message.content is None:
                    raise ModelError("tool results require a call ID and content")
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
            elif message.role == MessageRole.ASSISTANT:
                raise ModelError(
                    "Astra assistant history requires its original Responses items"
                )
            else:
                inputs.append(
                    {"role": message.role.value, "content": message.content or ""}
                )
        return {
            "model": self.model.name_or_path,
            "input": inputs,
            "tools": [
                {
                    "type": "function",
                    **JSON_OBJECT.validate_python(
                        tool.function.model_dump(mode="json")
                    ),
                    "strict": False,
                }
                for tool in request.tools
            ],
            "tool_choice": "required" if request.tools else "none",
            "parallel_tool_calls": False,
            "reasoning": {
                "effort": self.inference.reasoning_effort.value,
                **(
                    {"summary": self.inference.reasoning_summary}
                    if self.inference.reasoning_summary is not None
                    else {}
                ),
            },
            "max_output_tokens": self.inference.max_tokens,
            "store": False,
            "truncation": "disabled",
        }

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Count the full input, then generate once; seeds are not an Astra API parameter."""
        body = self.request_body(request)
        count_body = {
            key: value
            for key, value in body.items()
            if key not in {"store", "max_output_tokens"}
        }
        try:
            counted = InputTokenCount.model_validate(
                self.transport.request("responses/input_tokens", count_body)
            )
        except ValidationError as error:
            raise ModelError("OpenAI returned an invalid input token count") from error
        if (
            counted.input_tokens > ASTRA_INPUT_LIMIT
            or counted.input_tokens + self.inference.max_tokens
            > self.model.context_length
        ):
            raise ContextBudgetError(
                counted.input_tokens,
                self.inference.max_tokens,
                self.model.context_length,
            )
        raw = self.transport.request("responses", body)
        try:
            reply = ResponsesReply.model_validate(raw)
            items = [ResponseItem.model_validate(item) for item in reply.output]
        except ValidationError as error:
            raise ModelError("OpenAI returned an invalid Responses reply") from error
        if reply.model != self.model.name_or_path:
            raise ModelError("OpenAI returned a different model than requested")
        if reply.status not in {"completed", "incomplete"}:
            raise ModelError(f"OpenAI response did not complete: {reply.status}")
        calls: list[ToolCall] = []
        text: list[str] = []
        reasoning: list[str] = []
        for item in items:
            if item.type == "function_call":
                if item.name is None or item.call_id is None or item.arguments is None:
                    raise ModelError(
                        "OpenAI returned an incomplete function-call record"
                    )
                calls.append(
                    ToolCall(
                        id=item.call_id,
                        function=FunctionCall(name=item.name, arguments=item.arguments),
                    )
                )
            elif item.type in {"message", "reasoning"}:
                for part in item.content if item.type == "message" else item.summary:
                    value = part.get("text", part.get("refusal"))
                    if isinstance(value, str):
                        (text if item.type == "message" else reasoning).append(value)
            else:
                raise ModelError(
                    f"OpenAI returned an unsupported output item: {item.type}"
                )
        reason = (reply.incomplete_details or {}).get("reason")
        finish_reason = (
            str(reason)
            if reply.status == "incomplete"
            else "tool_calls"
            if calls
            else "stop"
        )
        usage = reply.usage
        return GenerationResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="".join(text),
                tool_calls=calls or None,
                reasoning_content="".join(reasoning) or None,
                continuation=ResponsesContinuation(
                    model=reply.model, output=reply.output
                ),
            ),
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens if usage else None,
                completion_tokens=usage.output_tokens if usage else None,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens
                if usage and usage.output_tokens_details
                else None,
                cached_tokens=usage.input_tokens_details.cached_tokens
                if usage and usage.input_tokens_details
                else None,
            ),
            finish_reason=finish_reason,
            truncated=reply.status == "incomplete" and reason == "max_output_tokens",
            prompt_tokens=counted.input_tokens,
            requested_max_tokens=self.inference.max_tokens,
            request=body,
            raw_response=raw,
        )


def model_client(model: ModelSettings, inference: InferenceSettings) -> ModelClient:
    """Select a supported connection without starting a server or making an API call."""
    if model.provider == ModelProvider.LOCAL and isinstance(
        inference, LocalInferenceSettings
    ):
        return LocalModelClient(model, inference)
    if model.provider == ModelProvider.OPENAI and isinstance(
        inference, AstraInferenceSettings
    ):
        return AstraModelClient(model, inference)
    raise ValueError("inference settings do not match the model provider")
