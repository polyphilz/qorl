"""Errors raised by model connections and context-budget checks."""


class ModelError(RuntimeError):
    """The model request failed independently of the agent's plan decision."""


class ModelRequestError(ModelError):
    """A non-rate-limit client error; retrying the same request cannot fix it."""


class TransientModelError(ModelError):
    """A transport failure or retryable HTTP response, with an optional retry delay."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class ContextBudgetError(ModelError):
    """The rendered history and requested completion do not fit the context."""

    def __init__(
        self, prompt_tokens: int, max_tokens: int, context_length: int
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.context_length = context_length
        super().__init__(
            f"model context budget exceeded: prompt={prompt_tokens} "
            f"completion={max_tokens} context={context_length}"
        )
