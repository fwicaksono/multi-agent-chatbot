"""Embedding provider interface and registry.

Embeddings are configured globally (EMBEDDING_PROVIDER / EMBEDDING_MODEL),
independently from each agent's chat provider: the vectors stored in the index
and the query vector must come from the same model, so switching an agent's
chat LLM must never invalidate its knowledge base.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from app.core.config import Settings, get_settings
from app.core.errors import AppError


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    prompt_tokens: int
    model: str
    provider: str


class EmbeddingProvider(ABC):
    name: ClassVar[str]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts in batches; the provider-specific call is `_embed_batch`."""
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            batch_vectors, batch_tokens = await self._embed_batch(batch)
            vectors.extend(batch_vectors)
            tokens += batch_tokens
        return EmbeddingResult(vectors, tokens, self.model, self.name)

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass


_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def register_embedding_provider(cls: type[EmbeddingProvider]) -> type[EmbeddingProvider]:
    _REGISTRY[cls.name] = cls
    return cls


@lru_cache
def _discover() -> None:
    import app.embeddings.providers as pkg

    for module in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{module.name}")


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    _discover()
    try:
        return _REGISTRY[settings.embedding_provider](settings)
    except KeyError:
        raise AppError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Available: {', '.join(sorted(_REGISTRY))}.",
            code="unknown_embedding_provider",
        ) from None


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return create_embedding_provider(get_settings())
