"""Hybrid search: vector + lexical branches fused with RRF (Session 10).

The two branches run CONCURRENTLY with ``asyncio.gather``. An ``AsyncSession``
cannot run two statements at once, so each branch opens its own session from the
factory (two pooled connections for the duration of the search).

Failure policy: if either branch fails, the failure is logged with the branch
name and the error is re-raised — we never silently degrade to a single branch
(that would change the ranking without anyone noticing). The caller
(``pipeline.retrieve``) maps it to ``RetrievalError``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.retrieval.candidates import StructuralFilters
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.lexical_search import lexical_search
from app.generation.rag.retrieval.vector_search import vector_search
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


@dataclass(frozen=True)
class HybridSearchResult:
    """Fused candidate pool plus the per-branch sizes (for logs/diagnostics)."""

    chunks: list[RetrievedChunk]
    candidates_evaluated: int
    vector_hits: int
    lexical_hits: int


async def hybrid_search(
    session_factory: Callable[[], AsyncSession],
    store: ChunkStore,
    *,
    query_embedding: list[float],
    query_text: str,
    vector_limit: int,
    lexical_limit: int,
    pool_size: int,
    distance_threshold: float,
    rrf_k: int,
    filters: StructuralFilters | None = None,
) -> HybridSearchResult:
    """Run both branches, fuse them with RRF and return up to ``pool_size`` chunks.

    Every returned chunk carries its ``rrf_score``; chunks found by the vector
    branch keep their real cosine ``distance`` and, when the lexical branch also
    found them, their ``lexical_score``. Lexical-only chunks carry the sentinel
    distance (see ``candidates.NO_VECTOR_DISTANCE``).
    """
    filters = filters or StructuralFilters()

    async def _vector_branch() -> tuple[list[RetrievedChunk], int]:
        async with session_factory() as session:
            return await vector_search(
                session,
                store,
                query_embedding=query_embedding,
                limit=vector_limit,
                distance_threshold=distance_threshold,
                filters=filters,
            )

    async def _lexical_branch() -> list[RetrievedChunk]:
        async with session_factory() as session:
            return await lexical_search(
                session, store, query_text=query_text, limit=lexical_limit, filters=filters
            )

    # return_exceptions=True: wait for BOTH branches to settle (no orphaned task
    # still holding a connection), then decide what to raise.
    vector_outcome, lexical_outcome = await asyncio.gather(
        _vector_branch(), _lexical_branch(), return_exceptions=True
    )
    failure: BaseException | None = None
    for branch, outcome in (("vector", vector_outcome), ("lexical", lexical_outcome)):
        if isinstance(outcome, BaseException):
            log.error("hybrid_branch_failed", branch=branch, error_type=type(outcome).__name__)
            failure = failure or outcome
    if failure is not None:
        raise failure

    vector_chunks, candidates_evaluated = vector_outcome  # type: ignore[misc]
    lexical_chunks: list[RetrievedChunk] = lexical_outcome  # type: ignore[assignment]

    started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        [vector_chunks, lexical_chunks],
        k=rrf_k,
        limit=pool_size,
        key=lambda chunk: chunk.id,
    )
    # Fusion keeps the vector copy of a chunk found by both branches (it has the
    # real distance); carry the lexical score over so neither signal is lost.
    lexical_scores = {chunk.id: chunk.lexical_score for chunk in lexical_chunks}
    chunks = [
        chunk.model_copy(
            update={
                "rrf_score": score,
                "lexical_score": lexical_scores.get(chunk.id, chunk.lexical_score),
            }
        )
        for chunk, score in fused
    ]
    log.info(
        "retrieval_stage",
        stage="fusion",
        rrf_k=rrf_k,
        vector_in=len(vector_chunks),
        lexical_in=len(lexical_chunks),
        overlap=len({c.id for c in vector_chunks} & lexical_scores.keys()),
        fused_out=len(chunks),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return HybridSearchResult(
        chunks=chunks,
        candidates_evaluated=candidates_evaluated,
        vector_hits=len(vector_chunks),
        lexical_hits=len(lexical_chunks),
    )
