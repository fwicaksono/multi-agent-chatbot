"""Provider implementations against mocked HTTP (no network).

Each provider is exercised for: a normal completion, a streamed completion with
usage, and translation of vendor errors (bad key, rate limit, quota, outage,
timeout) into the shared error taxonomy with actionable messages.
"""

from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.llm.base import ChatMessage, ChatRequest, StreamDelta, StreamEnd
from app.llm.errors import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.providers.litellm_provider import LiteLLMProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.openrouter_provider import OpenRouterProvider
from app.llm.registry import available_providers

REQUEST = ChatRequest(
    model="gpt-4.1-mini",
    messages=[ChatMessage("system", "Be brief."), ChatMessage("user", "Hi")],
    temperature=0.1,
    max_tokens=50,
)


def completion_body(text="Hello!", model="gpt-4.1-mini", usage=None):
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def sse_body(pieces: list[str], usage: dict | None = None, model="gpt-4.1-mini") -> bytes:
    lines = []
    for i, piece in enumerate(pieces):
        chunk = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": "stop" if i == len(pieces) - 1 else None,
                }
            ],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    if usage:
        tail = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [],
            "usage": usage,
        }
        lines.append(f"data: {json.dumps(tail)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def mock_client(handler, base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def openai_provider(settings, handler) -> OpenAIProvider:
    client = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://api.openai.test/v1",
        http_client=mock_client(handler, "https://api.openai.test/v1"),
        max_retries=0,
    )
    return OpenAIProvider(settings, client=client)


def collect_text(events) -> str:
    return "".join(e.text for e in events if isinstance(e, StreamDelta))


# --------------------------------------------------------------------------- registry


def test_registry_discovers_all_three_providers():
    assert {"openai", "litellm", "openrouter"} <= set(available_providers())


def test_each_provider_declares_metadata():
    for cls in available_providers().values():
        assert cls.display_name and cls.required_env and cls.suggested_models


# --------------------------------------------------------------------------- OpenAI


async def test_openai_complete_sends_expected_payload(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion_body())

    resp = await openai_provider(settings, handler).complete(REQUEST)
    assert resp.content == "Hello!"
    assert resp.provider == "openai"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (12, 3)
    assert seen["model"] == "gpt-4.1-mini"
    assert seen["temperature"] == 0.1
    assert seen["max_completion_tokens"] == 50
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


async def test_openai_reasoning_models_omit_temperature(settings):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion_body(model="gpt-5-mini"))

    req = ChatRequest(model="gpt-5-mini", messages=REQUEST.messages, temperature=0.3)
    await openai_provider(settings, handler).complete(req)
    assert "temperature" not in seen


async def test_openai_stream_yields_text_then_usage(settings):
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            content=sse_body(
                ["Hel", "lo"],
                usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            ),
            headers={"content-type": "text/event-stream"},
        )

    events = [e async for e in openai_provider(settings, handler).stream(REQUEST)]
    assert collect_text(events) == "Hello"
    end = events[-1]
    assert isinstance(end, StreamEnd) and end.usage.total_tokens == 11


@pytest.mark.parametrize(
    "status,body,error_cls,fragment",
    [
        (
            401,
            {
                "error": {
                    "message": "Incorrect API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            ProviderAuthError,
            "OPENAI_API_KEY",
        ),
        (
            429,
            {
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
            ProviderQuotaError,
            "credits",
        ),
        (
            404,
            {"error": {"message": "The model does not exist", "code": "model_not_found"}},
            ModelNotFoundError,
            "not available",
        ),
        (503, {"error": {"message": "overloaded"}}, ProviderUnavailableError, "unavailable"),
    ],
)
async def test_openai_errors_are_translated(settings, status, body, error_cls, fragment):
    settings.llm_max_retries = 0
    provider = openai_provider(settings, lambda r: httpx.Response(status, json=body))
    with pytest.raises(error_cls) as exc:
        await provider.complete(REQUEST)
    assert fragment in exc.value.message


async def test_openai_rate_limit_is_retried_then_succeeds(settings):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(
                429,
                json={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
            )
        return httpx.Response(200, json=completion_body("after retry"))

    resp = await openai_provider(settings, handler).complete(REQUEST)
    assert resp.content == "after retry"
    assert calls["n"] == 3  # 1 call + 2 retries (llm_max_retries=2)


async def test_openai_rate_limit_gives_up_after_max_retries(settings):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "Rate limit", "code": "rate_limit"}})

    with pytest.raises(ProviderRateLimitError, match="rate limiting"):
        await openai_provider(settings, handler).complete(REQUEST)
    assert calls["n"] == settings.llm_max_retries + 1


async def test_openai_timeout_is_translated(settings):
    settings.llm_max_retries = 0

    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderTimeoutError, match="did not respond"):
        await openai_provider(settings, handler).complete(REQUEST)


async def test_missing_key_raises_not_configured(settings):
    settings.openai_api_key = None
    with pytest.raises(ProviderNotConfiguredError, match="OPENAI_API_KEY"):
        await OpenAIProvider(settings).complete(REQUEST)


# --------------------------------------------------------------------------- OpenRouter


async def test_openrouter_reports_cost_and_sends_attribution_headers(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00042}
        return httpx.Response(200, json=completion_body("Hi from OR", "openai/gpt-4.1-mini", usage))

    provider = OpenRouterProvider(
        settings,
        client=mock_client(handler, "https://openrouter.test/api/v1"),
    )
    resp = await provider.complete(ChatRequest("openai/gpt-4.1-mini", REQUEST.messages))
    assert resp.content == "Hi from OR"
    assert resp.usage.cost_usd == pytest.approx(0.00042)
    assert seen["body"]["usage"] == {"include": True}
    assert "x-title" in seen["headers"] and "http-referer" in seen["headers"]


async def test_openrouter_stream_with_mid_stream_error(settings):
    settings.llm_max_retries = 0
    body = (
        b": OPENROUTER PROCESSING\n\n"
        + sse_body(["partial"])[: -len(b"data: [DONE]\n\n")]
        + b'data: {"error": {"code": 502, "message": "Upstream provider error"}}\n\n'
    )

    def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenRouterProvider(settings, client=mock_client(handler, "https://or.test"))
    events = []
    with pytest.raises(ProviderUnavailableError):
        async for e in provider.stream(REQUEST):
            events.append(e)
    assert collect_text(events) == "partial"  # text already emitted is not duplicated


async def test_openrouter_402_is_quota_error(settings):
    settings.llm_max_retries = 0
    provider = OpenRouterProvider(
        settings,
        client=mock_client(
            lambda r: httpx.Response(
                402, json={"error": {"code": 402, "message": "Insufficient credits"}}
            ),
            "https://or.test",
        ),
    )
    with pytest.raises(ProviderQuotaError):
        await provider.complete(REQUEST)


async def test_openrouter_model_catalogue_includes_prices(settings):
    data = {
        "data": [
            {
                "id": "openai/gpt-4o-mini",
                "name": "GPT-4o mini",
                "context_length": 128000,
                "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            }
        ]
    }
    provider = OpenRouterProvider(
        settings, client=mock_client(lambda r: httpx.Response(200, json=data), "https://or.test")
    )
    [model] = await provider.list_models()
    assert model.id == "openai/gpt-4o-mini"
    assert model.input_price_per_mtok == pytest.approx(0.15)
    assert model.output_price_per_mtok == pytest.approx(0.6)


# --------------------------------------------------------------------------- LiteLLM


async def test_litellm_complete_uses_proxy_cost_header(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json=completion_body("Hi from LiteLLM"),
            headers={"x-litellm-response-cost": "0.0001"},
        )

    provider = LiteLLMProvider(settings, client=mock_client(handler, "http://litellm.test"))
    resp = await provider.complete(REQUEST)
    assert resp.provider == "litellm"
    assert resp.content == "Hi from LiteLLM"
    assert resp.usage.cost_usd == pytest.approx(0.0001)
    assert seen["path"] == "/v1/chat/completions"


async def test_litellm_stream(settings):
    def handler(request):
        return httpx.Response(
            200,
            content=sse_body(
                ["a", "b", "c"],
                usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = LiteLLMProvider(settings, client=mock_client(handler, "http://litellm.test"))
    events = [e async for e in provider.stream(REQUEST)]
    assert collect_text(events) == "abc"
    assert events[-1].usage.completion_tokens == 3


async def test_litellm_unreachable_proxy_has_actionable_message(settings):
    settings.llm_max_retries = 0

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    provider = LiteLLMProvider(settings, client=mock_client(handler, "http://litellm.test"))
    with pytest.raises(ProviderUnavailableError, match="litellm service running"):
        await provider.complete(REQUEST)


async def test_litellm_invalid_key(settings):
    settings.llm_max_retries = 0
    provider = LiteLLMProvider(
        settings,
        client=mock_client(
            lambda r: httpx.Response(401, json={"error": {"message": "Authentication Error"}}),
            "http://litellm.test",
        ),
    )
    with pytest.raises(ProviderAuthError, match="LITELLM_API_KEY"):
        await provider.complete(REQUEST)


async def test_litellm_is_independent_of_other_providers():
    # LiteLLM must not be implemented as a wrapper around another provider class.
    assert not issubclass(LiteLLMProvider, OpenAIProvider | OpenRouterProvider)
    assert LiteLLMProvider.__mro__[1].__name__ == "LLMProvider"


async def test_empty_choices_raise_a_clear_fallback_eligible_error(settings):
    from app.llm.errors import ProviderError

    settings.llm_max_retries = 0
    filtered = {**completion_body(), "choices": []}
    openai_p = openai_provider(settings, lambda r: httpx.Response(200, json=filtered))
    router_p = OpenRouterProvider(
        settings,
        client=mock_client(lambda r: httpx.Response(200, json=filtered), "https://or.test"),
    )
    litellm_p = LiteLLMProvider(
        settings, client=mock_client(lambda r: httpx.Response(200, json=filtered), "http://l.test")
    )
    for provider in (openai_p, router_p, litellm_p):
        with pytest.raises(ProviderError, match="empty response") as exc:
            await provider.complete(REQUEST)
        assert exc.value.fallback_eligible and not exc.value.retryable
