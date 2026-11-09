"""Recall-then-rerank retrieval pipeline (Session 10).

The single ``retrieve()`` entrypoint composes the four configurations the exercise
measures, behind two switches the caller resolves (param → runtime → settings):

* ``search_mode="vector"`` — dense k-NN only (the Session 9 behaviour).
* ``search_mode="hybrid"`` — dense + lexical branches fused with RRF.
* ``rerank=False`` — keep the top ``top_k`` of the recall ordering.
* ``rerank=True``  — recall WIDE (``recall_k``, e.g. 50) then let the cross-encoder
  rescore and keep the top ``rerank_top_n`` (e.g. 5). The recall stage's only job
  is not to lose the relevant document; the reranker's job is to float it up.

It returns the same :class:`RetrievalResult` as ``search_chunks`` so every existing
consumer (orchestrator, stage endpoints) is unaffected.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.candidates import StructuralFilters, row_to_chunk
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.lexical_search import lexical_search
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk

log = structlog.get_logger()


async def retrieve(
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: str = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = 60,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker=None,
) -> RetrievalResult:
    """Run hybrid/vector retrieval with optional cross-encoder reranking.

    Parameters mirror ``search_chunks`` plus the Session 10 knobs. ``query_text``
    is required for the lexical branch and the reranker even when ``search_mode``
    is ``"vector"`` (the reranker scores against the raw query). ``reranker`` is
    injectable for tests; when ``None`` and ``rerank`` is True it is pulled from
    the composition root.

    Returns the top results best-first (cross-encoder order when reranking, else
    distance/RRF order). ``low_confidence`` is True (soft-fail) when nothing was
    retrieved at all — the orchestrator then short-circuits to an
    insufficient-context estimate instead of grounding on noise.
    """
    from app.dependencies import get_async_session_factory, get_chunk_store

    session_factory = get_async_session_factory()
    store = get_chunk_store()
    started = time.perf_counter()

    # Recall wide whenever a later stage (fusion or rerank) will re-sort; recall
    # exactly top_k only for the plain vector path, where the order is final.
    wide = rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else top_k

    try:
        async with session_factory() as session:
            vector_rows, candidates_evaluated = await store.search_filtered(
                session,
                query_vector=query_embedding,
                top_k=vector_limit,
                distance_threshold=distance_threshold,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )
            lexical_chunks: list[RetrievedChunk] = []
            if search_mode == "hybrid":
                lexical_chunks = await lexical_search(
                    session,
                    store,
                    query_text=query_text,
                    limit=recall_k,
                    filters=StructuralFilters(
                        sectors=sectors,
                        project_year_min=project_year_min,
                        project_year_max=project_year_max,
                        chunk_types=chunk_types,
                    ),
                )
    except Exception as exc:  # noqa: BLE001 — DB/connection failure.
        # Only the error type: DB error messages can echo bound parameters (the
        # user's query text), which must not reach the logs.
        log.error("rag_hybrid_search_failed", error_type=type(exc).__name__)
        raise RetrievalError("Vector store query failed.") from exc

    # Build the candidate pool once (id → chunk); vector distance wins over the
    # lexical sentinel when an id is in both branches.
    candidates: dict[int, RetrievedChunk] = {}
    for row in vector_rows:
        candidates[row.id] = row_to_chunk(row, distance=float(row.distance))
    for chunk in lexical_chunks:
        if chunk.id in candidates:
            candidates[chunk.id] = candidates[chunk.id].model_copy(
                update={"lexical_score": chunk.lexical_score}
            )
        else:
            candidates[chunk.id] = chunk

    # Recall ordering: RRF for hybrid, raw distance order for vector.
    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [[row.id for row in vector_rows], [chunk.id for chunk in lexical_chunks]],
            k=rrf_k,
        )
        ordered = [candidates[cid] for cid, _score in fused]
    else:
        ordered = [candidates[row.id] for row in vector_rows]

    if rerank and ordered:
        if reranker is None:
            from app.dependencies import get_reranker

            reranker = get_reranker()
        recall_pool = ordered[:recall_k]
        final = await asyncio.to_thread(
            reranker.rerank, query_text, recall_pool, top_n=rerank_top_n
        )
    else:
        final = ordered[:top_k]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rag_retrieve_done",
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_chunks),
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        search_time_ms=elapsed_ms,
    )
    return RetrievalResult(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
    )
