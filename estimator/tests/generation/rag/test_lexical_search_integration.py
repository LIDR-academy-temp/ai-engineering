"""Integration tests for the lexical branch against a real Postgres + pgvector.

The suite has no database fixtures, so these tests run only when
``TEST_DATABASE_URL`` points to a migrated database (``alembic upgrade head``),
e.g. the docker-compose one from the host::

    TEST_DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5433/<db> \\
        uv run pytest tests/generation/rag/test_lexical_search_integration.py

Every test inserts its own fixture chunks inside a transaction that is ROLLED
BACK at the end, filtered by a dedicated ``chunk_type`` — the real corpus is
never read nor modified.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.generation.rag.retrieval.candidates import StructuralFilters
from app.generation.rag.retrieval.lexical_search import lexical_search
from app.generation.rag.store.models import ChunkRow, DocumentRow
from app.generation.rag.store.repository import ChunkStore

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)

TEST_CHUNK_TYPE = "s10_lexical_test"
FILTERS = StructuralFilters(chunk_types=[TEST_CHUNK_TYPE])

FIXTURE_TEXTS = {
    "psd2": "Open banking gateway compliant with PSD2 and strong customer authentication.",
    "invoice": "Automated invoice generation for a single invoice per customer order.",
    "invoices": "Monthly invoices reconciliation across several warehouses.",
    "fhir": "Clinical records exchanged through HL7 FHIR resources.",
}


@pytest.fixture
async def session():
    engine = create_async_engine(DATABASE_URL)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        async with AsyncSession(bind=connection, expire_on_commit=False) as db_session:
            document = DocumentRow(
                source_path="tests://s10-lexical", document_type="s10_test", metadata_={}
            )
            db_session.add(document)
            await db_session.flush()
            db_session.add_all(
                ChunkRow(
                    document_id=document.id,
                    chunk_type=TEST_CHUNK_TYPE,
                    content=text,
                    metadata_={"budget_id": key},
                )
                for key, text in FIXTURE_TEXTS.items()
            )
            await db_session.flush()
            yield db_session
        await transaction.rollback()
    await engine.dispose()


async def _budget_ids(session, query: str) -> list[str]:
    chunks = await lexical_search(
        session, ChunkStore(), query_text=query, limit=10, filters=FILTERS
    )
    return [chunk.budget_id for chunk in chunks]


async def test_finds_a_chunk_by_an_exact_term(session):
    assert await _budget_ids(session, "PSD2") == ["psd2"]
    assert await _budget_ids(session, "FHIR") == ["fhir"]


async def test_stemming_matches_singular_and_plural(session):
    singular = set(await _budget_ids(session, "invoice"))
    plural = set(await _budget_ids(session, "invoices"))
    assert singular == plural == {"invoice", "invoices"}


async def test_any_term_matches_and_more_coverage_ranks_higher(session):
    ids = await _budget_ids(session, "PSD2 banking gateway invoice")
    assert ids[0] == "psd2"  # covers three terms
    assert {"invoice", "invoices"} <= set(ids)  # one term is enough to match


async def test_stopwords_only_query_returns_empty(session):
    assert await _budget_ids(session, "the and of with") == []


@pytest.mark.parametrize(
    "hostile",
    [
        "'; DROP TABLE chunks; --",
        '"unterminated quote',
        "a & | ! ( ) <-> :* \\ '",
        "PSD2' OR '1'='1",
    ],
)
async def test_special_characters_and_injection_do_not_break_the_query(session, hostile):
    ids = await _budget_ids(session, hostile)  # must not raise
    assert isinstance(ids, list)
    # The fixture rows are still there: nothing was executed but the search.
    assert await _budget_ids(session, "PSD2") == ["psd2"]
