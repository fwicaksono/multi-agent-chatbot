from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import settings_dep
from app.core.config import Settings
from app.db.session import get_db
from app.ingestion.extractors.ocr import tesseract_available

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    session: AsyncSession = Depends(get_db, scope="function"),
    settings: Settings = Depends(settings_dep),
) -> JSONResponse:
    """Liveness + readiness: checks the database and reports optional features."""
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    body = {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "ocr": "available" if settings.ocr_enabled and tesseract_available() else "unavailable",
        "environment": settings.environment,
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)
