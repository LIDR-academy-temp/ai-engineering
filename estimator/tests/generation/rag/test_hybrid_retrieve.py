"""Unit tests for the Session 10 recall-then-rerank pipeline (``retrieve``).

The vector store, lexical branch and cross-encoder are all faked: no Postgres,
no torch. We assert the four configurations (vector/hybrid × rerank on/off)
compose correctly — recall width, RRF fusion of the two branches, the
reranker reordering + top-N cut, and the soft-fail contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.dependencies as deps
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.pipeline import retrieve


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _factory():
    return FakeSession()


def _row(chunk_id: int, *, distance: float = 0.3, budget_id: str | None = None):
    return SimpleNamespace(
        id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"chunk-{chunk_id}",
        metadata_={
            "client_sector": "ecommerce",
            "year": 2024,
            "budget_id": budget_id or f"BUD-{chunk_id}",
        },
        distance=distance,
        rank=0.5,  # ts_rank_cd score when the row comes from the lexical branch
    )


class FakeStore:
    def __init__(self, vector_rows, lexical_rows, candidates=12):
        self.vector_rows = vector_rows
        self.lexical_rows = lexical_rows
        self.candidates = candidates
        self.vector_calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    async def search_filtered(self, session, **kwargs):
        self.vector_calls.append(kwargs)
        return self.vector_rows[: kwargs["top_k"]], self.candidates

    async def search_lexical(self, session, **kwargs):
        self.lexical_calls.append(kwargs)
        return self.lexical_rows[: kwargs["top_k"]]


class FakeReranker:
    """Reorders candidates by a content→score map and cuts to top_n."""

    model_name = "fake-cross-encoder"

    def __init__(self, scores_by_content):
        self.scores_by_content = scores_by_content
        self.calls: list[int] = []

    def rerank_with_scores(self, query, candidates, *, top_n, text_of=lambda c: c.content):
        self.calls.append(len(candidates))
        scored = [(c, self.scores_by_content[text_of(c)]) for c in candidates]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_n]


@pytest.fixture
def wire(monkeypatch):
    def _wire(store):
        monkeypatch.setattr(deps, "get_async_session_factory", lambda: _factory)
        monkeypatch.setattr(deps, "get_chunk_store", lambda: store)

    return _wire


async def test_vector_mode_no_rerank_returns_distance_order(wire):
    store = FakeStore(vector_rows=[_row(1), _row(2), _row(3)], lexical_rows=[])
    wire(store)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="vector",
        rerank=False,
        top_k=2,
    )

    assert [c.id for c in result.chunks] == [1, 2]
    assert result.low_confidence is False
    assert result.candidates_evaluated == 12
    assert result.chunks[0].budget_id == "BUD-1"
    # Pure vector / no rerank recalls exactly top_k (no wide recall).
    assert store.vector_calls[0]["top_k"] == 2
    assert store.lexical_calls == []  # lexical branch not touched


async def test_hybrid_mode_fuses_vector_and_lexical(wire):
    # Vector finds 1,2; lexical finds 3 (only the keyword branch surfaces it).
    store = FakeStore(
        vector_rows=[_row(1), _row(2)],
        lexical_rows=[_row(3), _row(1)],
    )
    wire(store)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
        recall_k=50,
    )

    ids = {c.id for c in result.chunks}
    assert ids == {1, 2, 3}  # lexical-only id 3 made it into the fused result
    # Hybrid recalls wide on both branches.
    assert store.vector_calls[0]["top_k"] == 50
    assert store.lexical_calls[0]["top_k"] == 50


async def test_rerank_reorders_and_cuts_to_top_n(wire):
    store = FakeStore(
        vector_rows=[_row(1), _row(2), _row(3)],
        lexical_rows=[],
    )
    wire(store)
    # Make chunk-3 the most relevant per the cross-encoder, chunk-1 the least.
    reranker = FakeReranker({"chunk-1": 0.1, "chunk-2": 0.5, "chunk-3": 0.9})

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="vector",
        rerank=True,
        recall_k=50,
        rerank_top_n=2,
        reranker=reranker,
    )

    assert [c.id for c in result.chunks] == [3, 2]  # reranked order, cut to 2


async def test_soft_fail_when_nothing_retrieved(wire):
    store = FakeStore(vector_rows=[], lexical_rows=[], candidates=7)
    wire(store)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=False,
    )

    assert result.chunks == []
    assert result.low_confidence is True
    assert result.candidates_evaluated == 7


class ExplodingReranker:
    """A reranker that must never be called."""

    model_name = "must-not-run"

    def rerank_with_scores(self, *args, **kwargs):
        raise AssertionError("reranker called with rerank=False")


class FailingStore(FakeStore):
    async def search_lexical(self, session, **kwargs):
        raise ConnectionError("lexical branch down")


async def test_rerank_disabled_returns_top_k_without_calling_reranker(wire):
    store = FakeStore(vector_rows=[_row(i) for i in range(1, 8)], lexical_rows=[_row(9)])
    wire(store)

    for mode in ("vector", "hybrid"):
        result = await retrieve(
            query_embedding=[0.0] * 1536,
            query_text="ecommerce checkout",
            search_mode=mode,
            rerank=False,
            top_k=5,
            reranker=ExplodingReranker(),
        )
        assert len(result.chunks) == 5
        assert all(c.rerank_score is None for c in result.chunks)


async def test_rerank_enabled_scores_the_whole_pool_and_returns_top_n(wire):
    rows = [_row(i, distance=0.1 + i / 100) for i in range(1, 51)]
    store = FakeStore(vector_rows=rows, lexical_rows=[])
    wire(store)
    # Reverse the recall order: the cross-encoder prefers the last candidates.
    reranker = FakeReranker({f"chunk-{i}": float(i) for i in range(1, 51)})

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="vector",
        rerank=True,
        top_k=5,
        recall_k=50,
        rerank_top_n=5,
        reranker=reranker,
    )

    assert reranker.calls == [50]  # the full candidate pool reached the reranker
    assert store.vector_calls[0]["top_k"] == 50  # recall wide before reranking
    assert [c.id for c in result.chunks] == [50, 49, 48, 47, 46]
    assert [c.rerank_score for c in result.chunks] == [50.0, 49.0, 48.0, 47.0, 46.0]


async def test_hybrid_rerank_pool_is_the_fused_ranking_capped_at_recall_k(wire):
    store = FakeStore(
        vector_rows=[_row(i) for i in range(1, 31)],
        lexical_rows=[_row(i) for i in range(21, 51)],
    )
    wire(store)
    reranker = FakeReranker({f"chunk-{i}": float(i) for i in range(1, 51)})

    await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=True,
        recall_k=40,
        rerank_top_n=5,
        reranker=reranker,
    )

    assert reranker.calls == [40]  # 50 distinct fused ids, capped at the pool size


async def test_hybrid_invokes_both_branches_and_exposes_scores(wire):
    store = FakeStore(vector_rows=[_row(1), _row(2)], lexical_rows=[_row(2), _row(3)])
    wire(store)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
        vector_candidate_limit=7,
        lexical_candidate_limit=9,
    )

    assert len(store.vector_calls) == 1 and len(store.lexical_calls) == 1
    assert store.vector_calls[0]["top_k"] == 7
    assert store.lexical_calls[0]["top_k"] == 9
    by_id = {c.id: c for c in result.chunks}
    assert result.chunks[0].id == 2  # found by both branches → fused first
    assert by_id[2].distance == pytest.approx(0.3)  # keeps the real vector distance
    assert by_id[2].lexical_score == pytest.approx(0.5)  # and the lexical score
    assert by_id[1].lexical_score is None  # vector-only
    assert by_id[3].distance == pytest.approx(1.0)  # lexical-only sentinel
    assert all(c.rrf_score is not None for c in result.chunks)


async def test_vector_mode_has_no_fusion_scores(wire):
    store = FakeStore(vector_rows=[_row(1)], lexical_rows=[_row(1)])
    wire(store)
    result = await retrieve(
        query_embedding=[0.0] * 1536, query_text="ecommerce checkout", search_mode="vector"
    )
    assert result.chunks[0].rrf_score is None
    assert result.chunks[0].lexical_score is None


async def test_a_failing_branch_is_raised_not_silently_dropped(wire):
    wire(FailingStore(vector_rows=[_row(1)], lexical_rows=[]))
    with pytest.raises(RetrievalError):
        await retrieve(
            query_embedding=[0.0] * 1536,
            query_text="ecommerce checkout",
            search_mode="hybrid",
        )


async def test_unknown_search_mode_is_rejected(wire):
    wire(FakeStore(vector_rows=[], lexical_rows=[]))
    with pytest.raises(ValueError):
        await retrieve(
            query_embedding=[0.0] * 1536, query_text="ecommerce checkout", search_mode="bm25"
        )
