"""Dense (vector) retrieval branch (Session 10 wrapper over the Session 9 k-NN).

The SQL lives in ``ChunkStore.search_filtered`` (cosine distance over the halfvec
cast, structural pre-filters, relevance threshold). This module times the call,
maps rows to the shared :class:`RetrievedChunk` type and logs the stage, so the
vector branch has the same shape as the lexical one and both compose in
``hybrid_search`` / ``pipeline``.
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.retrieval.candidates import StructuralFilters, row_to_chunk
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


async def vector_search(
    session: AsyncSession,
    store: ChunkStore,
    *,
    query_embedding: list[float],
    limit: int,
    distance_threshold: float,
    filters: StructuralFilters | None = None,
) -> tuple[list[RetrievedChunk], int]:
    """Return ``(chunks, candidates_evaluated)``, chunks by ascending distance.

    ``candidates_evaluated`` is how many chunks matched the structural filters
    before the threshold/limit (the Session 9 soft-fail diagnostic).
    """
    filters = filters or StructuralFilters()
    started = time.perf_counter()
    try:
        rows, candidates_evaluated = await store.search_filtered(
            session,
            query_vector=query_embedding,
            top_k=limit,
            distance_threshold=distance_threshold,
            **filters.as_kwargs(),
        )
    except Exception as exc:
        log.error(
            "retrieval_stage_failed",
            stage="vector_search",
            error_type=type(exc).__name__,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    chunks = [row_to_chunk(row, distance=float(row.distance)) for row in rows]
    log.info(
        "retrieval_stage",
        stage="vector_search",
        limit=limit,
        candidates=len(chunks),
        candidates_evaluated=candidates_evaluated,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return chunks, candidates_evaluated
