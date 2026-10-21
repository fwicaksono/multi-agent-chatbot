"""Application error types.

`AppError` subclasses carry an HTTP status, a stable machine-readable `code`
and a human-readable, actionable `message`. They are rendered by a single
exception handler into the envelope:

    {"error": {"code": "...", "message": "...", "details": {...}}}
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_error"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
