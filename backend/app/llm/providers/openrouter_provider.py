"""OpenRouter provider.

Calls the OpenRouter REST API directly with httpx. OpenRouter-specific
behaviour handled here:

* attribution headers (`HTTP-Referer`, `X-Title`) recommended by OpenRouter,
* `usage: {include: true}` so responses report the real cost in USD,
* errors delivered inside the SSE stream after it has started,
* a public model catalogue with per-token pricing (`GET /models`).
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

ENV_HINT = "OPENROUTER_API_KEY"


@register_provider
class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    display_name = "OpenRouter"
    required_env = ("OPENROUTER_API_KEY",)
    suggested_models = (
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
    )

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(settings)
        self._client = client

    def is_configured(self) -> bool:
        return self._client is not None or self.settings.openrouter_api_key is not None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.openrouter_base_url.rstrip("/"),
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {
            "HTTP-Referer": self.settings.openrouter_app_url,
            "X-Title": self.settings.openrouter_app_name,
        }
        if key := self.settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {key.get_secret_value()}"
        return headers

    def _payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
            "usage": {"include": True},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        return payload

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        try:
            resp = await self.client.post(
                "/chat/completions", json=self._payload(request, False), headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise timeout_error(self.display_name, request.model, self.timeout) from exc
        except httpx.TransportError as exc:
            raise unavailable_error(self.display_name, request.model, type(exc).__name__) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp, request.model)

        body = resp.json()
        # OpenRouter may return HTTP 200 with an error object (e.g. upstream failure).
        if err := body.get("error"):
            raise self._body_error(err, request.model)
        choice = body["choices"][0]
        return ChatResponse(
            content=choice["message"].get("content") or "",
            provider=self.name,
            model=body.get("model") or request.model,
            usage=_usage(body.get("usage")),
            finish_reason=choice.get("finish_reason"),
        )

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        usage = TokenUsage()
        finish_reason: str | None = None
        model = request.model
        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=self._payload(request, True),
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._http_error(resp, request.model)
                async for chunk in iter_sse_json(resp):
                    if err := chunk.get("error"):
                        raise self._body_error(err, request.model)
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
            raise unavailable_error(self.display_name, request.model, type(exc).__name__) from exc
        yield StreamEnd(provider=self.name, model=model, usage=usage, finish_reason=finish_reason)

    async def list_models(self) -> list[ModelInfo]:
        # The catalogue is public, so this works even without an API key.
        try:
            resp = await self.client.get("/models", headers=self._headers())
        except httpx.TransportError as exc:
            raise unavailable_error(self.display_name, None, type(exc).__name__) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp, None)
        models = []
        for m in resp.json().get("data", []):
            pricing = m.get("pricing") or {}
            models.append(
                ModelInfo(
                    id=m["id"],
                    name=m.get("name"),
                    context_length=m.get("context_length"),
                    input_price_per_mtok=_per_mtok(pricing.get("prompt")),
                    output_price_per_mtok=_per_mtok(pricing.get("completion")),
                )
            )
        return sorted(models, key=lambda m: m.id)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _http_error(self, resp: httpx.Response, model: str | None) -> ProviderError:
        return error_from_status(
            self.display_name,
            model,
            resp.status_code,
            error_detail(resp),
            ENV_HINT,
            retry_after_seconds(resp),
        )

    def _body_error(self, err: Any, model: str | None) -> ProviderError:
        err = err if isinstance(err, dict) else {"message": str(err)}
        code = err.get("code")
        status = code if isinstance(code, int) else 502
        return error_from_status(
            self.display_name, model, status, str(err.get("message", "")), ENV_HINT
        )


def _usage(raw: dict[str, Any] | None) -> TokenUsage:
    raw = raw or {}
    cost = raw.get("cost")
    return TokenUsage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        cost_usd=float(cost) if cost is not None else None,
    )


def _per_mtok(per_token: str | None) -> float | None:
    try:
        return round(float(per_token) * 1_000_000, 4) if per_token is not None else None
    except ValueError:
        return None
