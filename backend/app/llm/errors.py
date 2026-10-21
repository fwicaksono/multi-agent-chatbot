"""Provider-agnostic LLM error taxonomy.

Each provider implementation translates its own SDK/HTTP failures into one of
these classes, so the chat service, retry logic and fallback router never need
to know which vendor raised the error.

`retryable`         -> worth retrying the *same* provider after a back-off.
`fallback_eligible` -> worth trying the *next* provider in the fallback chain.
"""

from __future__ import annotations

from app.core.errors import AppError


class ProviderError(AppError):
    status_code = 502
    code = "provider_error"
    retryable = False
    fallback_eligible = True

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        retry_after: float | None = None,
        upstream_status: int | None = None,
    ) -> None:
        details = {"provider": provider, "model": model}
        if upstream_status is not None:
            details["upstream_status"] = upstream_status
        super().__init__(message, details=details)
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.upstream_status = upstream_status


class ProviderNotConfiguredError(ProviderError):
    status_code = 503
    code = "provider_not_configured"


class ProviderAuthError(ProviderError):
    status_code = 502
    code = "provider_auth_failed"


class ProviderQuotaError(ProviderError):
    status_code = 402
    code = "provider_quota_exceeded"


class ProviderRateLimitError(ProviderError):
    status_code = 429
    code = "provider_rate_limited"
    retryable = True


class ProviderTimeoutError(ProviderError):
    status_code = 504
    code = "provider_timeout"
    retryable = True


class ProviderUnavailableError(ProviderError):
    status_code = 503
    code = "provider_unavailable"
    retryable = True


class ModelNotFoundError(ProviderError):
    status_code = 400
    code = "model_not_found"


class ProviderBadRequestError(ProviderError):
    """The request itself was rejected; another provider would likely reject it too."""

    status_code = 400
    code = "provider_bad_request"
    fallback_eligible = False


class UnknownProviderError(AppError):
    status_code = 422
    code = "unknown_provider"


def _hint(provider: str, env_var: str) -> str:
    return f"Check {env_var} in your .env file and restart the services."


def auth_error(provider: str, model: str | None, env_var: str, status: int | None = None):
    return ProviderAuthError(
        f"{provider} rejected the API key (invalid, revoked or lacking permission). "
        + _hint(provider, env_var),
        provider=provider,
        model=model,
        upstream_status=status,
    )


def rate_limit_error(provider: str, model: str | None, retry_after: float | None = None):
    wait = f" Retry in ~{int(retry_after)}s." if retry_after else ""
    return ProviderRateLimitError(
        f"{provider} is rate limiting requests for model '{model}'.{wait} "
        "Lower the request rate, upgrade your plan, or configure a fallback provider.",
        provider=provider,
        model=model,
        retry_after=retry_after,
        upstream_status=429,
    )


def quota_error(provider: str, model: str | None, status: int | None = None):
    return ProviderQuotaError(
        f"{provider} reports insufficient quota/credits. Add credits or billing "
        "to the account, or switch this agent to another provider.",
        provider=provider,
        model=model,
        upstream_status=status,
    )


def timeout_error(provider: str, model: str | None, timeout: float):
    return ProviderTimeoutError(
        f"{provider} did not respond within {timeout:.0f}s. The service may be "
        "overloaded; try again, pick a faster model, or raise LLM_TIMEOUT_SECONDS.",
        provider=provider,
        model=model,
    )


def unavailable_error(provider: str, model: str | None, detail: str, status: int | None = None):
    return ProviderUnavailableError(
        f"{provider} is currently unavailable ({detail}). Try again shortly or "
        "configure a fallback provider.",
        provider=provider,
        model=model,
        upstream_status=status,
    )


def model_not_found_error(provider: str, model: str | None, status: int | None = None):
    return ModelNotFoundError(
        f"Model '{model}' is not available on {provider}. Pick a model from the "
        "provider's model list in the agent settings.",
        provider=provider,
        model=model,
        upstream_status=status,
    )


def bad_request_error(provider: str, model: str | None, detail: str, status: int | None = None):
    return ProviderBadRequestError(
        f"{provider} rejected the request: {detail}",
        provider=provider,
        model=model,
        upstream_status=status,
    )


def error_from_status(
    provider: str,
    model: str | None,
    status: int,
    detail: str,
    env_var: str,
    retry_after: float | None = None,
) -> ProviderError:
    """Map an HTTP status from an OpenAI-compatible API to our taxonomy."""
    lowered = detail.lower()
    if status in (401, 403):
        return auth_error(provider, model, env_var, status)
    if status == 402 or "insufficient_quota" in lowered or "insufficient credits" in lowered:
        return quota_error(provider, model, status)
    if status == 429:
        return rate_limit_error(provider, model, retry_after)
    if status == 404 or "model_not_found" in lowered or "no endpoints found" in lowered:
        return model_not_found_error(provider, model, status)
    if status in (408, 504):
        return ProviderTimeoutError(
            f"{provider} timed out upstream (HTTP {status}). Try again shortly.",
            provider=provider,
            model=model,
            upstream_status=status,
        )
    if status >= 500:
        return unavailable_error(provider, model, f"HTTP {status}", status)
    return bad_request_error(provider, model, detail[:300] or f"HTTP {status}", status)
