from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from typing import Any, Protocol


class ModelError(RuntimeError):
    pass


class ModelRequestError(ModelError):
    """A non-retryable HTTP request rejected by the model provider."""


class ModelClient(Protocol):
    def models(self) -> dict[str, Any]: ...

    def version(self) -> dict[str, Any]: ...

    def chat(self, body: dict[str, Any]) -> dict[str, Any]: ...


class OpenAIModelClient:
    """Tiny client for the subset of the OpenAI-compatible API we use."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self.api_key or os.environ.get("QORL_MODEL_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            urllib.parse.urljoin(f"{self.base_url}/", path),
            data=data,
            headers=headers,
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2_000]
            error_type = (
                ModelRequestError
                if HTTPStatus.BAD_REQUEST
                <= error.code
                < HTTPStatus.INTERNAL_SERVER_ERROR
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
                else ModelError
            )
            raise error_type(
                f"model server returned HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ModelError(f"model server request failed: {error}") from error

    def models(self) -> dict[str, Any]:
        return self.request("models")

    def version(self) -> dict[str, Any]:
        return self.request("../version")

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("chat/completions", body)
