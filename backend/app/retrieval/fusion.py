"""Reciprocal Rank Fusion (RRF).

RRF merges ranked lists using only ranks: score(d) = sum(1 / (k + rank)).
It needs no score normalisation, which matters here because cosine similarity
and ts_rank live on unrelated scales.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.repositories.chunks import ChunkHit


@dataclass(slots=True)
class FusedHit:
    hit: ChunkHit
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_similarity: float | None = None
    keyword_score: float | None = None
    matched_by: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    vector_hits: Sequence[ChunkHit], keyword_hits: Sequence[ChunkHit], k: int = 60
) -> list[FusedHit]:
    fused: dict[uuid.UUID, FusedHit] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        entry = fused.setdefault(hit.chunk_id, FusedHit(hit=hit, score=0.0))
        entry.score += 1.0 / (k + rank)
        entry.vector_rank = rank
        entry.vector_similarity = hit.score
        entry.matched_by.append("vector")

    for rank, hit in enumerate(keyword_hits, start=1):
        entry = fused.setdefault(hit.chunk_id, FusedHit(hit=hit, score=0.0))
        entry.score += 1.0 / (k + rank)
        entry.keyword_rank = rank
        entry.keyword_score = hit.score
        entry.matched_by.append("keyword")

    return sorted(fused.values(), key=lambda f: f.score, reverse=True)
