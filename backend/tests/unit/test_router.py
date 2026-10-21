"""Fallback router + the extensibility guarantee (a 4th provider plugs in cleanly)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.llm import registry
from app.llm.base import (
    ChatMessage,
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
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from app.llm.router import FallbackEvent, LLMRouter, Target, build_chain

REQ = ChatRequest(model="ignored", messages=[ChatMessage("user", "hi")])


class ScriptedProvider(LLMProvider):
    """Test double whose behaviour is scripted per instance."""

    name = "scripted"
    display_name = "Scripted"
    required_env = ("NONE",)
    suggested_models = ("m",)

    def __init__(self, settings, label: str, error: Exception | None = None, fail_after: int = 0):
        super().__init__(settings)
        self.label = label
        self.error = error
        self.fail_after = fail_after
        self.calls = 0

    def is_configured(self) -> bool:
        return True

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResponse(
            f"{self.label}:{request.model}", self.label, request.model, TokenUsage(1, 1)
        )

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        for i, word in enumerate(["one ", "two ", "three"]):
            if self.error and i == self.fail_after:
                raise self.error
            yield StreamDelta(word)
        yield StreamEnd(self.label, request.model, TokenUsage(1, 3))

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("m")]


class FakeFactory:
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers

    def get(self, name: str) -> LLMProvider:
        return self.providers[name]


@pytest.fixture
def no_retry(settings):
    settings.llm_max_retries = 0
    return settings


def test_build_chain_orders_and_deduplicates():
    chain = build_chain(
        Target("openai", "a"),
        Target("litellm", "b"),
        [("openai", "a"), ("openrouter", "c"), ("litellm", "b")],
    )
    assert chain == [Target("openai", "a"), Target("litellm", "b"), Target("openrouter", "c")]
    assert build_chain(Target("openai", "a"), None, [("x", "y")], enabled=False) == [
        Target("openai", "a")
    ]


async def test_complete_falls_back_on_rate_limit(no_retry):
    primary = ScriptedProvider(no_retry, "p", ProviderRateLimitError("slow", provider="p"))
    backup = ScriptedProvider(no_retry, "b")
    router = LLMRouter(FakeFactory({"p": primary, "b": backup}))

    resp, events = await router.complete(REQ, [Target("p", "m1"), Target("b", "m2")])
    assert resp.content == "b:m2"
    assert [(e.failed.provider, e.next.provider, e.error_code) for e in events] == [
        ("p", "b", "provider_rate_limited")
    ]


async def test_complete_does_not_fall_back_on_bad_request(no_retry):
    primary = ScriptedProvider(no_retry, "p", ProviderBadRequestError("bad", provider="p"))
    backup = ScriptedProvider(no_retry, "b")
    router = LLMRouter(FakeFactory({"p": primary, "b": backup}))
    with pytest.raises(ProviderBadRequestError):
        await router.complete(REQ, [Target("p", "m"), Target("b", "m")])
    assert backup.calls == 0


async def test_last_error_surfaces_when_chain_exhausted(no_retry):
    err = ProviderUnavailableError("down", provider="x")
    router = LLMRouter(
        FakeFactory(
            {"a": ScriptedProvider(no_retry, "a", err), "b": ScriptedProvider(no_retry, "b", err)}
        )
    )
    with pytest.raises(ProviderUnavailableError):
        await router.complete(REQ, [Target("a", "m"), Target("b", "m")])


async def test_stream_falls_back_before_first_token(no_retry):
    primary = ScriptedProvider(
        no_retry, "p", ProviderUnavailableError("down", provider="p"), fail_after=0
    )
    backup = ScriptedProvider(no_retry, "b")
    router = LLMRouter(FakeFactory({"p": primary, "b": backup}))

    events = [e async for e in router.stream(REQ, [Target("p", "m"), Target("b", "m")])]
    assert isinstance(events[0], FallbackEvent)
    assert "".join(e.text for e in events if isinstance(e, StreamDelta)) == "one two three"
    assert isinstance(events[-1], StreamEnd) and events[-1].provider == "b"


async def test_stream_never_splices_answers_after_first_token(no_retry):
    primary = ScriptedProvider(
        no_retry, "p", ProviderUnavailableError("down", provider="p"), fail_after=2
    )
    backup = ScriptedProvider(no_retry, "b")
    router = LLMRouter(FakeFactory({"p": primary, "b": backup}))

    received = []
    with pytest.raises(ProviderUnavailableError):
        async for e in router.stream(REQ, [Target("p", "m"), Target("b", "m")]):
            received.append(e)
    assert backup.calls == 0
    assert "".join(e.text for e in received if isinstance(e, StreamDelta)) == "one two "


async def test_provider_retries_stream_only_before_output(settings):
    settings.llm_max_retries = 2
    flaky = ScriptedProvider(
        settings, "f", ProviderRateLimitError("slow", provider="f"), fail_after=0
    )
    with pytest.raises(ProviderRateLimitError):
        _ = [e async for e in flaky.stream(REQ)]
    assert flaky.calls == 3  # initial + 2 retries, all failing before any token


async def test_fourth_provider_plugs_in_without_touching_core(settings, monkeypatch):
    """Adding a provider = one class + @register_provider. Router/chat code is untouched."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

    @registry.register_provider
    class EchoProvider(ScriptedProvider):
        name = "echo"
        display_name = "Echo"

        def __init__(self, settings):
            super().__init__(settings, "echo")

    assert "echo" in registry.available_providers()
    router = LLMRouter(registry.ProviderFactory(settings))
    resp, _ = await router.complete(REQ, [Target("echo", "any-model")])
    assert resp.content == "echo:any-model"


async def test_unknown_provider_in_chain_falls_through(settings):
    backup = ScriptedProvider(settings, "b")

    class Factory(FakeFactory):
        def get(self, name):
            if name == "nope":
                return registry.provider_class("nope")(settings)
            return super().get(name)

    router = LLMRouter(Factory({"b": backup}))
    resp, events = await router.complete(REQ, [Target("nope", "m"), Target("b", "m")])
    assert resp.content == "b:m" and events[0].error_code == "unknown_provider"
