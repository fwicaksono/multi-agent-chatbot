# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Backend image: FastAPI API and ingestion worker (same image, different cmd)
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PATH="/opt/venv/bin:$PATH"

# Tesseract (+ English and Indonesian language packs) for OCR of scanned PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Pre-fetch the tokenizer so containers never need internet access for it.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads \
    && chown -R appuser:appuser /data /app \
    && chmod +x /usr/local/bin/entrypoint.sh
USER appuser

ENV UPLOAD_DIR=/data/uploads
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["api"]
