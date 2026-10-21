from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import provider_factory_dep, settings_dep
from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.errors import ProviderError
from app.llm.registry import ProviderFactory, available_providers, provider_class
from app.schemas.api import ModelOut, ModelsResponse, ProviderOut, ProvidersResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("", response_model=ProvidersResponse)
async def list_providers(
    settings: Settings = Depends(settings_dep),
    factory: ProviderFactory = Depends(provider_factory_dep),
) -> ProvidersResponse:
    providers = [
        ProviderOut(
            name=name,
            display_name=cls.display_name,
            configured=factory.get(name).is_configured(),
            required_env=list(cls.required_env),
            suggested_models=list(cls.suggested_models),
        )
        for name, cls in available_providers().items()
    ]
    return ProvidersResponse(
        providers=providers,
        default_provider=settings.default_llm_provider,
        default_model=settings.default_llm_model,
        embedding={
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
        },
        fallback_chain=[f"{p}:{m}" for p, m in settings.fallback_chain],
    )


@router.get("/{name}/models", response_model=ModelsResponse)
async def list_models(
    name: str, factory: ProviderFactory = Depends(provider_factory_dep)
) -> ModelsResponse:
    """Live model list from the provider, falling back to curated suggestions."""
    cls = provider_class(name)
    suggested = [ModelOut(id=m) for m in cls.suggested_models]
    try:
        models = await factory.get(name).list_models()
    except ProviderError as exc:
        logger.info(
            "providers.model_list_unavailable", extra={"provider": name, "error_code": exc.code}
        )
        return ModelsResponse(
            provider=name, source="suggested", models=suggested, warning=exc.message
        )
    if not models:
        return ModelsResponse(provider=name, source="suggested", models=suggested)
    return ModelsResponse(
        provider=name,
        source="live",
        models=[ModelOut(**asdict(m)) for m in models],
    )
