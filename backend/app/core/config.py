"""Application configuration.

Every tunable value is read from environment variables (or a `.env` file) so the
application can be reconfigured without code changes. Secrets are typed as
`SecretStr` so they never leak into logs or reprs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
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
    fts_language: str = "english"

    # --- Chat --------------------------------------------------------------
    chat_history_messages: int = 10
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
