"""Application configuration.

Every tunable value is read from environment variables (or a `.env` file) so the
application can be reconfigured without code changes. Secrets are typed as
`SecretStr` so they never leak into logs or reprs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Multi-Agent RAG Chatbot"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # --- Database ----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot"
    db_pool_size: int = 10
    db_echo: bool = False

    # --- File storage & uploads -------------------------------------------
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 25
    max_files_per_upload: int = 20

    # --- Ingestion worker -------------------------------------------------
    # When true the API process also runs the ingestion worker loop. Handy for
    # local development; docker compose runs a dedicated worker container.
    embedded_worker: bool = False
    worker_concurrency: int = 2
    worker_poll_interval_seconds: float = 1.0
    worker_stale_lock_seconds: int = 600
    worker_max_attempts: int = 3

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = 450
    chunk_overlap_tokens: int = 60

    # --- OCR ---------------------------------------------------------------
    ocr_enabled: bool = True
    ocr_languages: str = "eng"
    ocr_dpi: int = 300
    ocr_min_chars_per_page: int = 25

    # --- Embeddings --------------------------------------------------------
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_candidate_k: int = 20
    retrieval_min_similarity: float = 0.2
    retrieval_rrf_k: int = 60
    # PostgreSQL text-search configurations for keyword search. Each chunk is
    # indexed with its detected language; queries match against all of them.
    fts_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["english", "indonesian"]
    )
    # Deprecated single-language setting, still honoured if FTS_LANGUAGES is unset.
    fts_language: str | None = None

    # --- Chat --------------------------------------------------------------
    chat_history_messages: int = Field(default=10, ge=0, le=500)
    # Older messages are condensed into a rolling summary that is always given
    # to the model, so long conversations stay remembered at a fixed cost.
    chat_summary_enabled: bool = True
    chat_summary_max_tokens: int = Field(default=400, ge=50, le=4000)
    query_rewrite_enabled: bool = True
    chat_max_output_tokens: int = 1024

    # --- LLM providers -----------------------------------------------------
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_retry_base_delay_seconds: float = 1.0
    # Comma separated "provider:model" pairs tried, in order, after an agent's
    # own primary (and optional fallback) provider fails.
    llm_fallback_chain: str = ""

    openai_api_key: SecretStr | None = None
    # Explicit default: the OpenAI SDK would otherwise read OPENAI_BASE_URL from
    # the process environment, where a blank value breaks every request.
    openai_base_url: str = "https://api.openai.com/v1"

    litellm_api_base: str = "http://localhost:4000"
    litellm_api_key: SecretStr | None = None
    # Someone else's LiteLLM proxy (e.g. your company's). When set it replaces the
    # bundled `litellm` container, both in Docker and when running locally.
    litellm_external_url: str | None = None

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_url: str = "http://localhost:8080"
    openrouter_app_name: str = "Multi-Agent RAG Chatbot"

    # Optional JSON object overriding/adding model prices (USD per 1M tokens):
    # {"my-model": {"input": 0.5, "output": 1.5}}
    model_pricing_overrides: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip().startswith("["):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("fts_languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        if isinstance(value, str):
            value = [v.strip().lower() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            for lang in value:
                if not re.fullmatch(r"[a-z_]+", str(lang)):
                    raise ValueError(f"invalid text-search configuration name: {lang!r}")
            if not value:
                raise ValueError("FTS_LANGUAGES must name at least one language")
        return value

    @field_validator("litellm_api_base", "litellm_external_url", mode="before")
    @classmethod
    def _normalise_litellm_url(cls, value: object) -> object:
        # Accept "https://host/", "https://host/v1" or "https://host/v1/": the
        # provider appends /v1/... itself.
        if isinstance(value, str):
            value = value.strip().rstrip("/")
            if value.endswith("/v1"):
                value = value[: -len("/v1")]
            return value or None
        return value

    @model_validator(mode="after")
    def _use_external_litellm(self) -> Settings:
        if self.litellm_external_url:
            self.litellm_api_base = self.litellm_external_url
        return self

    @model_validator(mode="after")
    def _legacy_fts_language(self) -> Settings:
        if self.fts_language and "fts_languages" not in self.model_fields_set:
            self.fts_languages = [self.fts_language.strip().lower()]
        return self

    @field_validator("openai_api_key", "litellm_api_key", "openrouter_api_key", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value: object) -> object:
        # Treat `OPENAI_API_KEY=` (blank in .env) as "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _default_openai_url(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "https://api.openai.com/v1"
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def fallback_chain(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for item in self.llm_fallback_chain.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            provider, model = item.split(":", 1)
            pairs.append((provider.strip(), model.strip()))
        return pairs


@lru_cache
def get_settings() -> Settings:
    return Settings()
