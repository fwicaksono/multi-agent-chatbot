"""Provider registry.

Providers register themselves with the `@register_provider` decorator. Every
module in `app/llm/providers/` is imported automatically, so adding a fourth
provider means adding one file there -- the chat and retrieval code look
providers up by name and never import a concrete implementation.
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.errors import UnknownProviderError

_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(cls: type[LLMProvider]) -> type[LLMProvider]:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise RuntimeError(f"LLM provider '{cls.name}' registered twice")
    _REGISTRY[cls.name] = cls
    return cls


@lru_cache
def _discover() -> None:
    import app.llm.providers as pkg

    for module in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{module.name}")


def available_providers() -> dict[str, type[LLMProvider]]:
    _discover()
    return dict(sorted(_REGISTRY.items()))


def provider_class(name: str) -> type[LLMProvider]:
    providers = available_providers()
    try:
        return providers[name]
    except KeyError:
        raise UnknownProviderError(
            f"Unknown LLM provider '{name}'. Available: {', '.join(providers)}.",
            details={"provider": name, "available": list(providers)},
        ) from None


class ProviderFactory:
    """Creates and caches one provider instance per name for a given Settings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._instances: dict[str, LLMProvider] = {}

    def get(self, name: str) -> LLMProvider:
        if name not in self._instances:
            self._instances[name] = provider_class(name)(self.settings)
        return self._instances[name]

    async def aclose(self) -> None:
        for provider in self._instances.values():
            await provider.aclose()
        self._instances.clear()


@lru_cache
def get_provider_factory() -> ProviderFactory:
    return ProviderFactory(get_settings())
