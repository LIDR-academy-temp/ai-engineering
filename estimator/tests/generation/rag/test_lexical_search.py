"""Unit tests for the lexical branch (no database).

Covers the OR-of-terms rewrite (the injection-safety boundary) and the mapping
done by ``lexical_search`` around a faked ``ChunkStore``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation.rag.retrieval.candidates import StructuralFilters
from app.generation.rag.retrieval.lexical_search import (
    MAX_QUERY_TERMS,
    lexical_search,
    to_any_terms_query,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mobile banking with PSD2", "Mobile or banking or with or PSD2"),
        ("HL7/FHIR integration", "HL7 or FHIR or integration"),
        # Quotes, negation and tsquery operators are stripped: only words remain.
        ('OAuth "payment gateway" -grocery', "OAuth or payment or gateway or grocery"),
        ("a & b | !c <-> (d):*", "a or b or c or d"),
        ("'; DROP TABLE chunks; --", "DROP or TABLE or chunks"),
        # Case-insensitive de-duplication keeps the first spelling.
        ("API api Api gateway", "API or gateway"),
        # A literal "or" from the user is dropped (it is our operator).
        ("cats or dogs OR birds", "cats or dogs or birds"),
    ],
)
def test_to_any_terms_query_keeps_only_word_tokens(raw, expected):
    assert to_any_terms_query(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", ":&|!()<->", "'\"\\--"])
def test_to_any_terms_query_without_words_is_empty(raw):
    assert to_any_terms_query(raw) == ""


def test_to_any_terms_query_caps_the_number_of_terms():
    raw = " ".join(f"term{i}" for i in range(MAX_QUERY_TERMS * 2))
    assert to_any_terms_query(raw).count(" or ") + 1 == MAX_QUERY_TERMS
    assert to_any_terms_query("a b c", max_terms=2) == "a or b"


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[dict] = []

    async def search_lexical(self, session, **kwargs):
        self.calls.append(kwargs)
        return self.rows[: kwargs["top_k"]]


def _row(chunk_id: int, rank: float):
    return SimpleNamespace(
        id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"chunk-{chunk_id}",
        metadata_={"client_sector": "finance", "year": 2024, "budget_id": f"BUD-{chunk_id}"},
        rank=rank,
    )


async def test_lexical_search_maps_rank_to_lexical_score_and_passes_filters():
    store = FakeStore([_row(1, 0.6), _row(2, 0.3)])
    filters = StructuralFilters(sectors=["finance"], chunk_types=["budget_component"])

    chunks = await lexical_search(
        object(), store, query_text="PSD2 compliance", limit=5, filters=filters
    )

    assert [c.id for c in chunks] == [1, 2]
    assert [c.lexical_score for c in chunks] == [0.6, 0.3]
    assert chunks[0].document_id == 10 and chunks[0].budget_id == "BUD-1"
    call = store.calls[0]
    assert call["query_text"] == "PSD2 or compliance"  # rewritten, never raw
    assert call["top_k"] == 5
    assert call["sectors"] == ["finance"]
    assert call["chunk_types"] == ["budget_component"]


async def test_lexical_search_without_words_skips_the_database():
    store = FakeStore([_row(1, 0.6)])
    assert await lexical_search(object(), store, query_text="!!! ---", limit=5) == []
    assert store.calls == []
