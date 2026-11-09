"""Recall-then-rerank retrieval pipeline (Session 10).

The single ``retrieve()`` entrypoint composes the four configurations the exercise
measures, behind two switches the caller resolves (param → runtime → settings):

* ``search_mode="vector"`` — dense k-NN only (the Session 9 behaviour).
* ``search_mode="hybrid"`` — dense + lexical branches fused with RRF.
* ``rerank=False`` — keep the top ``top_k`` of the recall ordering.
* ``rerank=True``  — recall WIDE (``recall_k``, e.g. 50) then let the cross-encoder
  rescore and keep the top ``rerank_top_n`` (e.g. 5). The recall stage's only job
  is not to lose the relevant document; the reranker's job is to float it up.

The stages live in their own modules (``vector_search``, ``lexical_search``,
``fusion``/``hybrid_search``, ``reranker``) and all speak ``RetrievedChunk``, so
this module only composes them. It returns the same :class:`RetrievalResult` as
``search_chunks`` so every existing consumer (orchestrator, stage endpoints) is
unaffected.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import structlog

from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.candidates import StructuralFilters
from app.generation.rag.retrieval.hybrid_search import hybrid_search
from app.generation.rag.retrieval.vector_search import vector_search
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk

log = structlog.get_logger()

SearchMode = Literal["vector", "hybrid"]


async def retrieve(
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: SearchMode = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = 60,
    vector_candidate_limit: int | None = None,
    lexical_candidate_limit: int | None = None,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker=None,
) -> RetrievalResult:
    """Run hybrid/vector retrieval with optional cross-encoder reranking.

    Parameters mirror ``search_chunks`` plus the Session 10 knobs:

    * ``recall_k`` — candidate pool size handed to the reranker (and the cap on
      the fused hybrid ranking).
    * ``vector_candidate_limit`` / ``lexical_candidate_limit`` — candidates per
      branch BEFORE fusion in hybrid mode (default: ``recall_k``).
    * ``query_text`` — required by the lexical branch and the reranker even in
      vector mode.
    * ``reranker`` — injectable for tests; when ``None`` and ``rerank`` is True it
      is pulled from the composition root.

    Returns the top results best-first (cross-encoder order when reranking, else
    distance/RRF order). ``low_confidence`` is True (soft-fail) when nothing was
    retrieved at all — the orchestrator then short-circuits to an
    insufficient-context estimate instead of grounding on noise.
    """
    if search_mode not in ("vector", "hybrid"):
        raise ValueError(f"Unknown search_mode: {search_mode!r}")

    from app.dependencies import get_async_session_factory, get_chunk_store

    session_factory = get_async_session_factory()
    store = get_chunk_store()
    filters = StructuralFilters(
        sectors=sectors,
        project_year_min=project_year_min,
        project_year_max=project_year_max,
        chunk_types=chunk_types,
    )
    started = time.perf_counter()

    vector_hits = lexical_hits = 0
    try:
        if search_mode == "hybrid":
            hybrid = await hybrid_search(
                session_factory,
                store,
                query_embedding=query_embedding,
                query_text=query_text,
                vector_limit=vector_candidate_limit or recall_k,
                lexical_limit=lexical_candidate_limit or recall_k,
                pool_size=recall_k,
                distance_threshold=distance_threshold,
                rrf_k=rrf_k,
                filters=filters,
            )
            pool: list[RetrievedChunk] = hybrid.chunks
            candidates_evaluated = hybrid.candidates_evaluated
            vector_hits, lexical_hits = hybrid.vector_hits, hybrid.lexical_hits
        else:
            # Recall wide only when the reranker will re-sort; otherwise the
            # distance order is final and exactly top_k is enough.
            async with session_factory() as session:
                pool, candidates_evaluated = await vector_search(
                    session,
                    store,
                    query_embedding=query_embedding,
                    limit=recall_k if rerank else top_k,
                    distance_threshold=distance_threshold,
                    filters=filters,
                )
            vector_hits = len(pool)
    except Exception as exc:  # noqa: BLE001 — DB/connection failure.
        # Only the error type: DB error messages can echo bound parameters (the
        # user's query text), which must not reach the logs.
        log.error(
            "rag_hybrid_search_failed", search_mode=search_mode, error_type=type(exc).__name__
        )
        raise RetrievalError("Vector store query failed.") from exc

    if rerank and pool:
        if reranker is None:
            from app.dependencies import get_reranker

            reranker = get_reranker()
        # The cross-encoder scores against the SAME text the search used
        # (``query_text``): in the estimate flow that is the reformulated,
        # corpus-aligned search text, not the raw transcript — scoring the
        # candidates against what they were recalled for keeps both stages aligned.
        final = await _rerank(reranker, query_text, pool[:recall_k], top_n=rerank_top_n)
    else:
        final = pool[:top_k]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rag_retrieve_done",
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=vector_hits,
        lexical_hits=lexical_hits,
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        search_time_ms=elapsed_ms,
    )
    return RetrievalResult(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
    )


async def _rerank(
    reranker, query_text: str, candidates: list[RetrievedChunk], *, top_n: int
) -> list[RetrievedChunk]:
    """Cross-encoder stage: rescore ``candidates`` and keep the best ``top_n``.

    Inference is synchronous and CPU-heavy, so it runs in a worker thread
    (``asyncio.to_thread``) to keep the event loop free for other requests.
    """
    model = getattr(reranker, "model_name", type(reranker).__name__)
    started = time.perf_counter()
    try:
        ranked = await asyncio.to_thread(
            reranker.rerank_with_scores, query_text, candidates, top_n=top_n
        )
    except Exception as exc:
        log.error(
            "retrieval_stage_failed",
            stage="rerank",
            model=model,
            error_type=type(exc).__name__,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    final = [chunk.model_copy(update={"rerank_score": score}) for chunk, score in ranked]
    log.info(
        "retrieval_stage",
        stage="rerank",
        model=model,
        candidates_in=len(candidates),
        candidates_out=len(final),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return final
