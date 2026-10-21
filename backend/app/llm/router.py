"""Fallback routing across providers.

`LLMRouter` walks an ordered chain of (provider, model) targets. Each provider
already retries transient errors itself; when it finally gives up with a
fallback-eligible error (timeout, rate limit, outage, bad key, missing
configuration...) the router moves on to the next target and reports the
switch so the UI can show which provider actually answered.

For streams, fallback only happens before the first token is emitted --
switching mid-answer would splice text from two different models.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import ChatRequest, ChatResponse, StreamDelta, StreamEnd, StreamEvent
from app.llm.errors import ProviderError, UnknownProviderError
from app.llm.registry import ProviderFactory

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Target:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(slots=True)
class FallbackEvent:
    """Emitted when the router abandons `failed` and tries `next`."""

    failed: Target
    next: Target
    error_code: str
    error_message: str

    def to_dict(self) -> dict:
        return {
            "failed_provider": self.failed.provider,
            "failed_model": self.failed.model,
            "next_provider": self.next.provider,
            "next_model": self.next.model,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


RouterStreamEvent = StreamEvent | FallbackEvent


def build_chain(
    primary: Target,
    agent_fallback: Target | None,
    global_chain: list[tuple[str, str]],
    enabled: bool = True,
) -> list[Target]:
    chain = [primary]
    if enabled:
        if agent_fallback:
            chain.append(agent_fallback)
        chain.extend(Target(p, m) for p, m in global_chain)
    # De-duplicate while keeping order.
    seen: set[Target] = set()
    return [t for t in chain if not (t in seen or seen.add(t))]


class LLMRouter:
    def __init__(self, factory: ProviderFactory) -> None:
        self.factory = factory

    def _with_target(self, request: ChatRequest, target: Target) -> ChatRequest:
        return ChatRequest(
            model=target.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    def _should_fallback(self, exc: Exception, index: int, chain: list[Target]) -> bool:
        if index >= len(chain) - 1:
            return False
        if isinstance(exc, UnknownProviderError):
            return True
        return isinstance(exc, ProviderError) and exc.fallback_eligible

    def _log_fallback(self, event: FallbackEvent) -> None:
        logger.warning(
            "llm.fallback",
            extra={
                "failed": str(event.failed),
                "next": str(event.next),
                "error_code": event.error_code,
            },
        )

    async def complete(
        self, request: ChatRequest, chain: list[Target]
    ) -> tuple[ChatResponse, list[FallbackEvent]]:
        events: list[FallbackEvent] = []
        for i, target in enumerate(chain):
            try:
                provider = self.factory.get(target.provider)
                return await provider.complete(self._with_target(request, target)), events
            except (ProviderError, UnknownProviderError) as exc:
                if not self._should_fallback(exc, i, chain):
                    raise
                event = FallbackEvent(target, chain[i + 1], exc.code, exc.message)
                self._log_fallback(event)
                events.append(event)
        raise RuntimeError("empty provider chain")  # pragma: no cover

    async def stream(
        self, request: ChatRequest, chain: list[Target]
    ) -> AsyncIterator[RouterStreamEvent]:
        for i, target in enumerate(chain):
            emitted = False
            try:
                provider = self.factory.get(target.provider)
                async for event in provider.stream(self._with_target(request, target)):
                    if isinstance(event, StreamDelta):
                        emitted = True
                    yield event
                    if isinstance(event, StreamEnd):
                        return
                return
            except (ProviderError, UnknownProviderError) as exc:
                if emitted or not self._should_fallback(exc, i, chain):
                    raise
                event = FallbackEvent(target, chain[i + 1], exc.code, exc.message)
                self._log_fallback(event)
                yield event
