"""Dependency wiring.

Long-lived collaborators (provider clients, embedder, services) are created
once per process and shared; FastAPI routes receive them via `Depends`, which
also makes them easy to override in tests.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider, get_embedding_provider
from app.ingestion.storage import FileStorage
from app.llm.registry import ProviderFactory, get_provider_factory
from app.llm.router import LLMRouter
from app.retrieval.retriever import HybridRetriever
from app.services.chat import ChatService


def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_storage() -> FileStorage:
    return FileStorage(get_settings().upload_dir)


def provider_factory_dep() -> ProviderFactory:
    return get_provider_factory()


def embedder_dep() -> EmbeddingProvider:
    return get_embedding_provider()


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_settings(), get_embedding_provider())


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(get_settings(), LLMRouter(get_provider_factory()), get_retriever())
