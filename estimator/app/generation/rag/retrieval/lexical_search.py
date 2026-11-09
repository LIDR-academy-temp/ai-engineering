"""Full-text (lexical) retrieval branch (Session 10).

Keyword search catches what dense embeddings blur: exact acronyms, product and
technology names (``PSD2``, ``HL7``, ``FHIR``, ``OAuth``). The SQL lives in
``ChunkStore.search_lexical`` (parametrised ``websearch_to_tsquery`` over the
generated ``content_tsv`` column); this module only times the call, maps rows to
the shared :class:`RetrievedChunk` type and logs the stage.

Logging never includes the query text — only sizes, limits and durations (the
``request_id`` is merged in from the structlog context by the API middleware).
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.retrieval.candidates import StructuralFilters, row_to_chunk
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


async def lexical_search(
    session: AsyncSession,
    store: ChunkStore,
    *,
    query_text: str,
    limit: int,
    filters: StructuralFilters | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``limit`` chunks ranked by lexical relevance, best first.

    Each chunk carries its ``ts_rank_cd`` score in ``lexical_score`` and the
    sentinel ``distance`` (it has no cosine distance from this branch). A query
    whose tsquery is empty (only stop-words) returns ``[]``.
    """
    filters = filters or StructuralFilters()
    started = time.perf_counter()
    try:
        rows = await store.search_lexical(
            session, query_text=query_text, top_k=limit, **filters.as_kwargs()
        )
    except Exception as exc:
        log.error(
            "retrieval_stage_failed",
            stage="lexical_search",
            error_type=type(exc).__name__,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    chunks = [row_to_chunk(row, lexical_score=float(row.rank)) for row in rows]
    log.info(
        "retrieval_stage",
        stage="lexical_search",
        limit=limit,
        candidates=len(chunks),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return chunks
