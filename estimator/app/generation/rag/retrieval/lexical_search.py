"""Full-text (lexical) retrieval branch (Session 10).

Keyword search catches what dense embeddings blur: exact acronyms, product and
technology names (``PSD2``, ``HL7``, ``FHIR``, ``OAuth``). The SQL lives in
``ChunkStore.search_lexical`` (parametrised ``websearch_to_tsquery`` over the
generated ``content_tsv`` column); this module only times the call, maps rows to
the shared :class:`RetrievedChunk` type and logs the stage.

Match semantics — ANY term, ranked by coverage. ``websearch_to_tsquery`` ANDs
plain words, so a full project description ("mobile banking app with OAuth2 and
PSD2 compliance") would only match a chunk containing EVERY word: in practice,
nothing, and hybrid search would silently equal vector search. We therefore keep
only the word tokens of the query and join them with ``or`` (web-search syntax
for OR) — ``ts_rank_cd`` then ranks chunks by how densely they cover the terms.
The rewritten string contains nothing but word tokens and ``or``, so the user
cannot inject tsquery operators, and it is still sent as a bound parameter.

Logging never includes the query text — only sizes, limits and durations (the
``request_id`` is merged in from the structlog context by the API middleware).
"""

from __future__ import annotations

import re
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.retrieval.candidates import StructuralFilters, row_to_chunk
from app.generation.rag.schemas import RetrievedChunk
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()

# Upper bound on OR-ed terms: keeps the tsquery (and its ranking cost) bounded no
# matter how long the input is. Query text is already length-capped upstream.
MAX_QUERY_TERMS = 64
_WORD = re.compile(r"\w+", re.UNICODE)


def to_any_terms_query(query_text: str, *, max_terms: int = MAX_QUERY_TERMS) -> str:
    """Rewrite free text into an OR-of-words web-search query.

    Keeps only word tokens (dropping quotes, ``-``, ``:``, ``&``… so no operator
    can be injected), de-duplicates them case-insensitively preserving order and
    caps them at ``max_terms``. Stop-words are left in: Postgres drops them when
    building the tsquery. Returns ``""`` when there is no word at all.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for token in _WORD.findall(query_text):
        lowered = token.lower()
        if lowered in seen or lowered == "or":
            continue
        seen.add(lowered)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return " or ".join(terms)


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
    with no word token, or whose tsquery is empty (only stop-words), returns
    ``[]``.
    """
    filters = filters or StructuralFilters()
    started = time.perf_counter()
    any_terms = to_any_terms_query(query_text)
    if not any_terms:
        log.info("retrieval_stage", stage="lexical_search", limit=limit, candidates=0, terms=0)
        return []
    try:
        rows = await store.search_lexical(
            session, query_text=any_terms, top_k=limit, **filters.as_kwargs()
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
        terms=any_terms.count(" or ") + 1,
        candidates=len(chunks),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return chunks
