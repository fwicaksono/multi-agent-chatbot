"""Structured logging.

Logs are emitted as one JSON object per line (or a readable console format for
local development). A request id is carried in a context variable so every log
line produced while handling a request can be correlated.

Usage:
    logger = get_logger(__name__)
    logger.info("document.processed", extra={"document_id": doc_id, "chunks": 12})
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes present on every LogRecord; anything else was passed via `extra=`.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
    "color_message",  # uvicorn's ANSI-coloured duplicate of the message
}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        if request_id := request_id_ctx.get():
            payload["request_id"] = request_id
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        extras = " ".join(f"{k}={v}" for k, v in _extra_fields(record).items())
        rid = request_id_ctx.get()
        line = f"{ts} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if rid:
            line += f" request_id={rid}"
        if extras:
            line += f" {extras}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    # Route uvicorn's loggers through the same handler; silence noisy libraries.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our middleware
    for noisy in ("httpx", "httpcore", "openai", "sqlalchemy.engine", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
