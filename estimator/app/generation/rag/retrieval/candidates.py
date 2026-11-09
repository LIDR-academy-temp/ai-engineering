"""Shared building blocks for the retrieval branches (Session 10).

* :class:`StructuralFilters` — the sector / year / chunk-type predicates every
  branch must apply identically, so vector and lexical see the same candidates.
* :func:`row_to_chunk` — maps a store row (vector or lexical) onto the typed
  :class:`RetrievedChunk` contract, so every branch returns the SAME type and the
  stages (fusion, rerank) compose without adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.generation.rag.schemas import RetrievedChunk

# Distance assigned to a candidate surfaced ONLY by the lexical branch: it never
# entered the vector ranking, so it has no cosine distance. 1.0 = "far" in cosine
# terms. It is a display/sentinel value — fusion and reranking ignore it, and the
# real lexical relevance lives in ``RetrievedChunk.lexical_score``.
NO_VECTOR_DISTANCE = 1.0


@dataclass(frozen=True)
class StructuralFilters:
    """Optional metadata filters; ``None`` means "do not filter on this axis"."""

    sectors: list[str] | None = None
    project_year_min: int | None = None
    project_year_max: int | None = None
    chunk_types: list[str] | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted by the ``ChunkStore`` search methods."""
        return {
            "sectors": self.sectors,
            "project_year_min": self.project_year_min,
            "project_year_max": self.project_year_max,
            "chunk_types": self.chunk_types,
        }


def row_to_chunk(
    row: Any,
    *,
    distance: float = NO_VECTOR_DISTANCE,
    lexical_score: float | None = None,
) -> RetrievedChunk:
    """Map a ``ChunkStore`` row onto the typed retrieval contract."""
    metadata = row.metadata_ or {}
    return RetrievedChunk(
        id=row.id,
        document_id=getattr(row, "document_id", None),
        content=row.content,
        sector=str(metadata.get("client_sector", "unknown")),
        project_year=int(metadata.get("year", 0)),
        chunk_type=row.chunk_type,
        distance=distance,
        budget_id=metadata.get("budget_id"),
        lexical_score=lexical_score,
    )
