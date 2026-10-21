"""Model price table for cost estimation.

Prices are USD per 1M tokens (list prices; verify against the vendor's
pricing page). When a provider reports the actual cost (OpenRouter, LiteLLM)
that value wins; this table is the fallback. Override or extend it without
code changes via MODEL_PRICING_OVERRIDES='{"model": {"input": x, "output": y}}'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Price:
    input: float
    output: float = 0.0


DEFAULT_PRICES: dict[str, Price] = {
    # OpenAI chat
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1-nano": Price(0.10, 0.40),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-5": Price(1.25, 10.00),
    "gpt-5-mini": Price(0.25, 2.00),
    "gpt-5-nano": Price(0.05, 0.40),
    "o4-mini": Price(1.10, 4.40),
    # OpenAI embeddings
    "text-embedding-3-small": Price(0.02),
    "text-embedding-3-large": Price(0.13),
    # Popular OpenRouter models (OpenRouter normally reports exact cost)
    "claude-sonnet-4.5": Price(3.00, 15.00),
    "gemini-2.5-flash": Price(0.30, 2.50),
    "llama-3.3-70b-instruct": Price(0.13, 0.40),
}


@lru_cache
def _price_table() -> dict[str, Price]:
    table = dict(DEFAULT_PRICES)
    raw = get_settings().model_pricing_overrides.strip()
    if raw:
        try:
            for model, p in json.loads(raw).items():
                table[model] = Price(float(p.get("input", 0)), float(p.get("output", 0)))
        except (ValueError, AttributeError, TypeError):
            logger.warning("pricing.invalid_overrides", extra={"value": raw[:200]})
    return table


def _normalise(model: str) -> str:
    # "openai/gpt-4o-mini" -> "gpt-4o-mini"; "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini"
    return model.split("/")[-1].lower()


def price_for(model: str) -> Price | None:
    table = _price_table()
    name = _normalise(model)
    if name in table:
        return table[name]
    # Dated snapshots ("gpt-4o-mini-2024-07-18"): longest matching prefix wins.
    for key in sorted(table, key=len, reverse=True):
        if name.startswith(key + "-"):
            return table[key]
    return None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000
