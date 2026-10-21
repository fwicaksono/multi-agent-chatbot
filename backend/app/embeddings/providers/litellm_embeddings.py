"""Embeddings through the LiteLLM proxy (`POST /v1/embeddings`).

Lets you route embeddings to any backend configured in litellm/config.yaml
(e.g. a self-hosted model) without code changes: EMBEDDING_PROVIDER=litellm.
"""

from __future__ import annotations

import httpx

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider, register_embedding_provider
from app.llm.errors import error_from_status, timeout_error, unavailable_error
from app.llm.sse import error_detail, retry_after_seconds

PROVIDER = "LiteLLM embeddings"


@register_embedding_provider
class LiteLLMEmbeddingProvider(EmbeddingProvider):
    name = "litellm"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(settings)
        if client is None:
            headers = {}
            if key := settings.litellm_api_key:
                headers["Authorization"] = f"Bearer {key.get_secret_value()}"
            client = httpx.AsyncClient(
                base_url=settings.litellm_api_base.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
            )
        self.client = client

    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        payload = {"model": self.model, "input": texts, "dimensions": self.dimensions}
        try:
            resp = await self.client.post("/v1/embeddings", json=payload)
        except httpx.TimeoutException as exc:
            raise timeout_error(PROVIDER, self.model, self.settings.llm_timeout_seconds) from exc
        except httpx.TransportError as exc:
            raise unavailable_error(PROVIDER, self.model, type(exc).__name__) from exc
        if resp.status_code >= 400:
            raise error_from_status(
                PROVIDER,
                self.model,
                resp.status_code,
                error_detail(resp),
                "LITELLM_API_BASE / LITELLM_API_KEY",
                retry_after_seconds(resp),
            )
        body = resp.json()
        data = sorted(body["data"], key=lambda d: d.get("index", 0))
        tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
        return [d["embedding"] for d in data], tokens

    async def aclose(self) -> None:
        await self.client.aclose()
