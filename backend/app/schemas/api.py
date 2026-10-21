"""Request/response models for the HTTP API."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Agents -------------------------------------------------------------------


class AgentBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    instructions: str = Field(default="", max_length=20000)
    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    fallback_provider: str | None = Field(default=None, max_length=50)
    fallback_model: str | None = Field(default=None, max_length=200)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @field_validator("name", "model", "provider")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("fallback_provider", "fallback_model")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return v.strip() or None if v is not None else None


class AgentCreate(AgentBase):
    pass


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    instructions: str | None = Field(default=None, max_length=20000)
    provider: str | None = Field(default=None, max_length=50)
    model: str | None = Field(default=None, max_length=200)
    fallback_provider: str | None = Field(default=None, max_length=50)
    fallback_model: str | None = Field(default=None, max_length=200)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class DocumentCountsOut(BaseModel):
    total: int = 0
    ready: int = 0
    processing: int = 0
    failed: int = 0


class AgentOut(ORMModel):
    id: uuid.UUID
    name: str
    description: str
    instructions: str
    provider: str
    model: str
    fallback_provider: str | None
    fallback_model: str | None
    temperature: float
    created_at: datetime
    updated_at: datetime
    document_counts: DocumentCountsOut = Field(default_factory=DocumentCountsOut)


# --- Documents ----------------------------------------------------------------


class DocumentOut(ORMModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    filename: str
    file_type: str
    size_bytes: int
    status: Literal["processing", "ready", "failed"]
    stage: str
    error_code: str | None
    error_message: str | None
    page_count: int | None
    ocr_page_count: int
    chunk_count: int
    char_count: int
    embedding_model: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None


class UploadRejection(BaseModel):
    filename: str
    code: str
    message: str


class UploadResult(BaseModel):
    accepted: list[DocumentOut]
    rejected: list[UploadRejection]


# --- Conversations & chat -----------------------------------------------------


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationOut(ORMModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageOut(ORMModel):
    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    status: str
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    provider: str | None
    model: str | None
    rewritten_query: str | None
    fallback_events: list[dict[str, Any]]
    error: dict[str, Any] | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_ms: int | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ChatRequestIn(BaseModel):
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


# --- Retrieval debug ----------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    sources: list[dict[str, Any]]


# --- Providers ----------------------------------------------------------------


class ProviderOut(BaseModel):
    name: str
    display_name: str
    configured: bool
    required_env: list[str]
    suggested_models: list[str]


class ModelOut(BaseModel):
    id: str
    name: str | None = None
    context_length: int | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


class ModelsResponse(BaseModel):
    provider: str
    source: Literal["live", "suggested"]
    models: list[ModelOut]
    warning: str | None = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderOut]
    default_provider: str
    default_model: str
    embedding: dict[str, Any]
    fallback_chain: list[str]


# --- Usage --------------------------------------------------------------------


class UsageTotals(BaseModel):
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class UsageByModel(UsageTotals):
    provider: str
    model: str
    kind: str


class UsageDaily(UsageTotals):
    day: date


class AgentUsageOut(BaseModel):
    agent_id: uuid.UUID
    totals: UsageTotals
    by_model: list[UsageByModel]
    daily: list[UsageDaily]
    unpriced_requests: int


class AgentUsageSummary(UsageTotals):
    agent_id: uuid.UUID
    agent_name: str
