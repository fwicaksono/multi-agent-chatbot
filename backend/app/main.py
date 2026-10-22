"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import agents, conversations, documents, health, providers, usage
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.embeddings.base import get_embedding_provider
from app.llm.registry import get_provider_factory
from app.repositories.chunks import verify_text_search_configs

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await verify_text_search_configs(settings.fts_languages)
    worker_task = None
    worker = None
    if settings.embedded_worker:
        from app.workers.ingestion import build_worker

        worker = build_worker(settings)
        worker_task = asyncio.create_task(worker.run())
    logger.info(
        "app.started",
        extra={"environment": settings.environment, "embedded_worker": settings.embedded_worker},
    )
    yield
    if worker and worker_task:
        worker.stop()
        await worker_task
    await get_provider_factory().aclose()
    await get_embedding_provider().aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = request_id
        if request.url.path != "/api/health":
            logger.info(
                "http.request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log("http.app_error", extra={"error_code": exc.code, "path": request.url.path})
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        message = "; ".join(f"{e['field']}: {e['message']}" for e in errors) or "Invalid request."
        return JSONResponse(
            {
                "error": {
                    "code": "validation_error",
                    "message": message,
                    "details": {"errors": errors},
                }
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("http.unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            {
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Check the server logs "
                    "(search for the x-request-id header value).",
                    "details": {},
                }
            },
            status_code=500,
        )

    for module in (health, providers, agents, documents, conversations, usage):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
