"""Turn `[n]` markers in an answer into structured citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.rag.prompts import location_label
from app.retrieval.fusion import FusedHit

# Matches [1], [1, 3], [1,2,5] and [1-3].
_MARKER = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
SNIPPET_CHARS = 600


def cited_indexes(answer: str, max_index: int) -> list[int]:
    found: list[int] = []
    for group in _MARKER.findall(answer):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[\-–]", part):
                lo, hi = (int(x) for x in re.split(r"\s*[\-–]\s*", part))
                numbers = range(lo, hi + 1) if hi - lo < 20 else []
            else:
                numbers = [int(part)]
            for n in numbers:
                if 1 <= n <= max_index and n not in found:
                    found.append(n)
    return found


def source_payload(index: int, hit: FusedHit) -> dict[str, Any]:
    h = hit.hit
    return {
        "index": index,
        "chunk_id": str(h.chunk_id),
        "document_id": str(h.document_id),
        "document_name": h.document_name,
        "file_type": h.file_type,
        "page_number": h.page_number,
        "section": h.section,
        "location": location_label(hit),
        "snippet": h.content[:SNIPPET_CHARS],
        "score": round(hit.score, 5),
        "vector_similarity": (
            round(hit.vector_similarity, 4) if hit.vector_similarity is not None else None
        ),
        "matched_by": hit.matched_by,
    }


def build_sources(hits: Sequence[FusedHit]) -> list[dict[str, Any]]:
    """All passages given to the model (what was *consulted*)."""
    return [source_payload(i, hit) for i, hit in enumerate(hits, start=1)]


def build_citations(answer: str, sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of sources the answer actually cites, in citation order."""
    by_index = {s["index"]: s for s in sources}
    return [by_index[i] for i in cited_indexes(answer, len(sources))]
