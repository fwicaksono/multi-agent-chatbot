"""Minimal Server-Sent-Events reader for HTTP streaming APIs.

This is a transport utility (parsing `data:` lines), not a provider; the
providers that speak HTTP directly use it to read streamed completions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


async def iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield each JSON `data:` payload until the `[DONE]` sentinel."""
    async for line in response.aiter_lines():
        line = line.strip()
        # Blank lines separate events; lines starting with ":" are comments
        # (OpenRouter sends ": OPENROUTER PROCESSING" keep-alives).
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def error_detail(response: httpx.Response) -> str:
    """Best-effort extraction of an error message from a JSON error body."""
    try:
        body = response.json()
    except (json.JSONDecodeError, httpx.ResponseNotRead, UnicodeDecodeError):
        return response.text[:300] if response.is_stream_consumed else ""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)[:300]
        return str(err)[:300]
    return str(body)[:300]
