"""Shared test fixtures.

Unit tests never touch the network or a database: providers are exercised
against `httpx.MockTransport`, and embeddings come from a deterministic fake.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "sample_data"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        openai_api_key="sk-test",
        openrouter_api_key="or-test",
        litellm_api_base="http://litellm.test",
        litellm_api_key="sk-litellm-test",
        llm_max_retries=2,
        llm_retry_base_delay_seconds=0,
        llm_timeout_seconds=5,
        embedding_dimensions=8,
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        ocr_enabled=False,
    )


class FakeEmbedder(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedder (no network)."""

    name = "fake"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls: list[list[str]] = []

    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dimensions] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors, sum(len(t.split()) for t in texts)


@pytest.fixture
def fake_embedder(settings: Settings) -> FakeEmbedder:
    return FakeEmbedder(settings)


@pytest.fixture
def sample_data() -> Path:
    return SAMPLE_DATA
