"""Hybrid retriever: dense vectors + Postgres full-text, fused with RRF.

Vector search finds paraphrases ("time off" ~ "annual leave"); keyword search
nails exact terms that embeddings blur (clause numbers, names, "Section 4.2").
Low-similarity vector-only candidates are dropped so an unrelated question
yields no context and the model answers "I don't know".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider, EmbeddingResult
from app.repositories import chunks as chunk_repo
from app.retrieval.fusion import FusedHit, reciprocal_rank_fusion

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass(slots=True)
class RetrievalResult:
    hits: list[FusedHit]
    query: str
    embedding: EmbeddingResult | None


class HybridRetriever:
    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        self.settings = settings
        self.embedder = embedder

    async def retrieve(
        self,
        session: AsyncSession,
        agent_id: uuid.UUID,
        query: str,
        *,
        top_k: int | None = None,
        mode: SearchMode = "hybrid",
    ) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        candidates = max(s.retrieval_candidate_k, top_k)

        vector_hits: list[chunk_repo.ChunkHit] = []
        keyword_hits: list[chunk_repo.ChunkHit] = []
        embedding: EmbeddingResult | None = None

        if mode in ("hybrid", "vector"):
            embedding = await self.embedder.embed([query])
            vector_hits = await chunk_repo.vector_search(
                session, agent_id, embedding.vectors[0], candidates
            )
            vector_hits = [h for h in vector_hits if h.score >= s.retrieval_min_similarity]

        if mode in ("hybrid", "keyword"):
            keyword_hits = await chunk_repo.keyword_search(
                session, agent_id, query, candidates, s.fts_languages
            )

        fused = reciprocal_rank_fusion(vector_hits, keyword_hits, k=s.retrieval_rrf_k)
        return RetrievalResult(hits=fused[:top_k], query=query, embedding=embedding)
