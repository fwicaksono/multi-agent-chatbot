"""OpenAI provider, implemented with the official `openai` SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ModelInfo,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    TokenUsage,
)
from app.llm.errors import (
    ProviderError,
    auth_error,
    bad_request_error,
    empty_response_error,
    model_not_found_error,
    quota_error,
    rate_limit_error,
    timeout_error,
    unavailable_error,
)
from app.llm.registry import register_provider

# Reasoning models reject `temperature`; we simply omit it for them.
# (Models not listed here that also reject it are detected and remembered at
# runtime by LLMProvider, so new models work without a code change.)
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
_CHAT_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


@register_provider
class OpenAIProvider(LLMProvider):
    name = "openai"
    display_name = "OpenAI"
    required_env = ("OPENAI_API_KEY",)
    suggested_models = (
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-6-luna",
    )

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        super().__init__(settings)
        self._client = client

    def is_configured(self) -> bool:
        return self._client is not None or self.settings.openai_api_key is not None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            key = self.settings.openai_api_key
            self._client = AsyncOpenAI(
                api_key=key.get_secret_value() if key else None,
                base_url=self.settings.openai_base_url,
                timeout=self.timeout,
                max_retries=0,  # retries are handled by LLMProvider
            )
        return self._client

    def _params(self, request: ChatRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.max_tokens:
            params["max_completion_tokens"] = request.max_tokens
        if request.temperature is not None and not _is_reasoning_model(request.model):
            params["temperature"] = request.temperature
        return params

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        try:
            resp = await self.client.chat.completions.create(**self._params(request))
        except openai.OpenAIError as exc:
            raise self._translate(exc, request.model) from exc
        if not resp.choices:
            raise empty_response_error(self.display_name, request.model)
        choice = resp.choices[0]
        usage = TokenUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return ChatResponse(
            content=choice.message.content or "",
            provider=self.name,
            model=resp.model or request.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        usage = TokenUsage()
        finish_reason: str | None = None
        model = request.model
        try:
            stream = await self.client.chat.completions.create(
                **self._params(request), stream=True, stream_options={"include_usage": True}
            )
            async with stream:  # closes the HTTP response even if the consumer stops early
                async for chunk in stream:
                    model = chunk.model or model
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    if choice.delta and choice.delta.content:
                        yield StreamDelta(choice.delta.content)
        except openai.OpenAIError as exc:
            raise self._translate(exc, request.model) from exc
        yield StreamEnd(provider=self.name, model=model, usage=usage, finish_reason=finish_reason)

    async def list_models(self) -> list[ModelInfo]:
        self._ensure_configured(None)
        try:
            page = await self.client.models.list()
        except openai.OpenAIError as exc:
            raise self._translate(exc, None) from exc
        ids = sorted(m.id for m in page.data if m.id.startswith(_CHAT_MODEL_PREFIXES))
        return [ModelInfo(id=i) for i in ids]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _translate(self, exc: openai.OpenAIError, model: str | None) -> ProviderError:
        if isinstance(exc, openai.APITimeoutError):
            return timeout_error(self.display_name, model, self.timeout)
        if isinstance(exc, openai.APIConnectionError):
            return unavailable_error(self.display_name, model, "connection failed")
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return auth_error(self.display_name, model, "OPENAI_API_KEY", exc.status_code)
        if isinstance(exc, openai.RateLimitError):
            if getattr(exc, "code", None) == "insufficient_quota" or "quota" in str(exc).lower():
                return quota_error(self.display_name, model, 429)
            retry_after = _retry_after(exc)
            return rate_limit_error(self.display_name, model, retry_after)
        if isinstance(exc, openai.NotFoundError):
            return model_not_found_error(self.display_name, model, 404)
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                return unavailable_error(
                    self.display_name, model, f"HTTP {exc.status_code}", exc.status_code
                )
            return bad_request_error(self.display_name, model, exc.message, exc.status_code)
        return unavailable_error(self.display_name, model, type(exc).__name__)


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value else None
    except (AttributeError, ValueError):
        return None
