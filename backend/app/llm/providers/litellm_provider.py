"""LiteLLM provider.

Talks to a LiteLLM Proxy (the LiteLLM gateway server) over HTTP. LiteLLM is a
provider in its own right here: it has its own endpoint, credentials (a proxy
master/virtual key), model aliases defined in `litellm/config.yaml`, and its own
cost reporting (`x-litellm-response-cost`). It does not reuse the OpenAI or
OpenRouter provider classes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

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
from app.llm.errors import ProviderError, error_from_status, timeout_error, unavailable_error
from app.llm.registry import register_provider
from app.llm.sse import error_detail, iter_sse_json, retry_after_seconds

ENV_HINT = "LITELLM_API_BASE / LITELLM_API_KEY"


@register_provider
class LiteLLMProvider(LLMProvider):
    name = "litellm"
    display_name = "LiteLLM"
    required_env = ("LITELLM_API_BASE", "LITELLM_API_KEY")
    suggested_models = ("gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1")

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(settings)
        self._client = client

    def is_configured(self) -> bool:
        return bool(self.settings.litellm_api_base)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if key := self.settings.litellm_api_key:
                headers["Authorization"] = f"Bearer {key.get_secret_value()}"
            self._client = httpx.AsyncClient(
                base_url=self.settings.litellm_api_base.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    def _payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        try:
            resp = await self.client.post(
                "/v1/chat/completions", json=self._payload(request, False)
            )
        except httpx.TimeoutException as exc:
            raise timeout_error(self.display_name, request.model, self.timeout) from exc
        except httpx.TransportError as exc:
            raise self._unreachable(request.model, exc) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp, request.model)

        body = resp.json()
        choice = body["choices"][0]
        usage = _usage(body.get("usage"))
        if cost := resp.headers.get("x-litellm-response-cost"):
            usage.cost_usd = _float(cost)
        return ChatResponse(
            content=choice["message"].get("content") or "",
            provider=self.name,
            model=body.get("model") or request.model,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
        )

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        usage = TokenUsage()
        finish_reason: str | None = None
        model = request.model
        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=self._payload(request, True)
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._http_error(resp, request.model)
                async for chunk in iter_sse_json(resp):
                    if err := chunk.get("error"):
                        raise _stream_error(self.display_name, request.model, err)
                    model = chunk.get("model") or model
                    if chunk.get("usage"):
                        usage = _usage(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield StreamDelta(text)
        except httpx.TimeoutException as exc:
            raise timeout_error(self.display_name, request.model, self.timeout) from exc
        except httpx.TransportError as exc:
            raise self._unreachable(request.model, exc) from exc
        yield StreamEnd(provider=self.name, model=model, usage=usage, finish_reason=finish_reason)

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self.client.get("/v1/models")
        except httpx.TransportError as exc:
            raise self._unreachable(None, exc) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp, None)
        ids = sorted({m["id"] for m in resp.json().get("data", []) if "*" not in m["id"]})
        return [ModelInfo(id=i) for i in ids]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _unreachable(self, model: str | None, exc: Exception) -> ProviderError:
        return unavailable_error(
            self.display_name,
            model,
            f"cannot reach the LiteLLM proxy at {self.settings.litellm_api_base}: "
            f"{type(exc).__name__}; is the litellm service running?",
        )

    def _http_error(self, resp: httpx.Response, model: str | None) -> ProviderError:
        return error_from_status(
            self.display_name,
            model,
            resp.status_code,
            error_detail(resp),
            ENV_HINT,
            retry_after_seconds(resp),
        )


def _usage(raw: dict[str, Any] | None) -> TokenUsage:
    raw = raw or {}
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
    )


def _stream_error(provider: str, model: str, err: Any) -> ProviderError:
    """Errors raised after streaming started arrive as an `error` object in a chunk."""
    err = err if isinstance(err, dict) else {"message": str(err)}
    code = str(err.get("code", ""))
    status = int(code) if code.isdigit() else 500
    return error_from_status(provider, model, status, str(err.get("message", "")), ENV_HINT)


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
