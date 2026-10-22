"""The provider interface shared by every chat LLM implementation.

The chat service only ever talks to `LLMProvider`. Each concrete provider lives
in its own module under `app/llm/providers/` and registers itself with
`@register_provider`; see `app/llm/registry.py`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, TypeVar

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.errors import ProviderBadRequestError, ProviderError, ProviderNotConfiguredError

logger = get_logger(__name__)

Role = Literal["system", "user", "assistant"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cost reported by the provider itself (OpenRouter, LiteLLM). When None the
    # usage tracker estimates cost from the pricing table.
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ChatResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


@dataclass(slots=True)
class StreamDelta:
    """A piece of generated text."""

    text: str


@dataclass(slots=True)
class StreamEnd:
    """Terminal stream event carrying usage information."""

    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


StreamEvent = StreamDelta | StreamEnd


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    name: str | None = None
    context_length: int | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


class LLMProvider(ABC):
    """A chat-completion backend.

    Subclasses set the class attributes below and implement `_complete`,
    `_stream` and `list_models`. Retries with exponential back-off for
    retryable errors are handled here so implementations stay small.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    # Environment variable(s) users must set; surfaced in the UI and errors.
    required_env: ClassVar[tuple[str, ...]] = ()
    suggested_models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout = settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self.retry_base_delay = settings.llm_retry_base_delay_seconds
        # Models that rejected `temperature` (many reasoning models only accept
        # the default). Learned at runtime so new models need no code change.
        self._no_temperature: set[str] = set()

    # -- To implement -------------------------------------------------------
    @abstractmethod
    def is_configured(self) -> bool:
        """True when the credentials/endpoint needed to call the provider are set."""

    @abstractmethod
    async def _complete(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Models available from the provider (may call the provider's API)."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources (HTTP clients)."""

    # -- Public API ---------------------------------------------------------
    async def complete(self, request: ChatRequest) -> ChatResponse:
        self._ensure_configured(request.model)
        request = self._adapt(request)
        try:
            return await self._with_retries(lambda: self._complete(request), request.model)
        except ProviderBadRequestError as exc:
            if not self._rejects_temperature(request, exc):
                raise
            request = self._adapt(request)
            return await self._with_retries(lambda: self._complete(request), request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream a completion.

        Retries are only attempted while nothing has been emitted yet; once the
        first token reaches the caller a failure is surfaced immediately so we
        never emit duplicated text.
        """
        self._ensure_configured(request.model)
        request = self._adapt(request)
        attempt = 0
        while True:
            emitted = False
            try:
                async for event in self._stream(request):
                    emitted = True
                    yield event
                return
            except ProviderBadRequestError as exc:
                if emitted or not self._rejects_temperature(request, exc):
                    raise
                request = self._adapt(request)  # retry once without temperature
            except ProviderError as exc:
                if emitted or not exc.retryable or attempt >= self.max_retries:
                    raise
                attempt += 1
                await self._sleep_before_retry(exc, attempt, request.model)

    def _adapt(self, request: ChatRequest) -> ChatRequest:
        if request.temperature is not None and request.model in self._no_temperature:
            return replace(request, temperature=None)
        return request

    def _rejects_temperature(self, request: ChatRequest, exc: ProviderBadRequestError) -> bool:
        """True (and remembered) when the model refused the temperature parameter."""
        if request.temperature is None or "temperature" not in exc.message.lower():
            return False
        self._no_temperature.add(request.model)
        logger.info(
            "llm.temperature_unsupported", extra={"provider": self.name, "model": request.model}
        )
        return True

    def _ensure_configured(self, model: str | None) -> None:
        if not self.is_configured():
            env = ", ".join(self.required_env) or "provider settings"
            raise ProviderNotConfiguredError(
                f"{self.display_name} is not configured. Set {env} in your .env file "
                "and restart the services, or choose another provider for this agent.",
                provider=self.name,
                model=model,
            )

    async def _with_retries(self, call: Callable[[], Awaitable[T]], model: str) -> T:
        attempt = 0
        while True:
            try:
                return await call()
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                attempt += 1
                await self._sleep_before_retry(exc, attempt, model)

    async def _sleep_before_retry(self, exc: ProviderError, attempt: int, model: str) -> None:
        delay = self.retry_base_delay * (2 ** (attempt - 1))
        if exc.retry_after:
            delay = max(delay, min(exc.retry_after, 30.0))
        logger.warning(
            "llm.retry",
            extra={
                "provider": self.name,
                "model": model,
                "attempt": attempt,
                "delay_s": round(delay, 2),
                "error_code": exc.code,
            },
        )
        await asyncio.sleep(delay)
