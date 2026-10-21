"""OpenAI embeddings (text-embedding-3-*)."""

from __future__ import annotations

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider, register_embedding_provider
from app.llm.errors import (
    ProviderError,
    ProviderNotConfiguredError,
    auth_error,
    bad_request_error,
    quota_error,
    rate_limit_error,
    timeout_error,
    unavailable_error,
)

PROVIDER = "OpenAI embeddings"


@register_embedding_provider
class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        super().__init__(settings)
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            key = self.settings.openai_api_key
            if key is None:
                raise ProviderNotConfiguredError(
                    "Embeddings use OpenAI but OPENAI_API_KEY is not set. Add it to .env "
                    "or set EMBEDDING_PROVIDER to another provider.",
                    provider=PROVIDER,
                    model=self.model,
                )
            self._client = AsyncOpenAI(
                api_key=key.get_secret_value(),
                base_url=self.settings.openai_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
        return self._client

    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        kwargs = {}
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions
        try:
            resp = await self.client.embeddings.create(model=self.model, input=texts, **kwargs)
        except openai.OpenAIError as exc:
            raise _translate(exc, self.model, self.settings.llm_timeout_seconds) from exc
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return vectors, resp.usage.prompt_tokens if resp.usage else 0

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def _translate(exc: openai.OpenAIError, model: str, timeout: float) -> ProviderError:
    if isinstance(exc, openai.APITimeoutError):
        return timeout_error(PROVIDER, model, timeout)
    if isinstance(exc, openai.APIConnectionError):
        return unavailable_error(PROVIDER, model, "connection failed")
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return auth_error(PROVIDER, model, "OPENAI_API_KEY", exc.status_code)
    if isinstance(exc, openai.RateLimitError):
        if "quota" in str(exc).lower():
            return quota_error(PROVIDER, model, 429)
        return rate_limit_error(PROVIDER, model)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return unavailable_error(PROVIDER, model, f"HTTP {exc.status_code}", exc.status_code)
        return bad_request_error(PROVIDER, model, exc.message, exc.status_code)
    return unavailable_error(PROVIDER, model, type(exc).__name__)
